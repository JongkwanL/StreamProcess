package main

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/grpc-ecosystem/grpc-gateway/v2/runtime"
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promhttp"
	"github.com/sirupsen/logrus"
	"github.com/spf13/cobra"
	"github.com/spf13/viper"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/reflection"

	pb "github.com/streamprocess/streamprocess/protos"
	"github.com/streamprocess/streamprocess/pkg/queue"
)

// StreamProcessServer implements the gRPC streaming services
type StreamProcessServer struct {
	pb.UnimplementedSTTServiceServer
	pb.UnimplementedOCRServiceServer
	
	consumer      *queue.RedisConsumer
	producer      *queue.RedisProducer
	logger        *logrus.Logger
	metrics       *ServerMetrics
}

// ServerMetrics for Prometheus monitoring
type ServerMetrics struct {
	RequestsTotal       *prometheus.CounterVec
	RequestDuration     *prometheus.HistogramVec
	ActiveStreams       prometheus.Gauge
	QueuedJobs          *prometheus.GaugeVec
	ProcessingErrors    *prometheus.CounterVec
}

// RedisProducer for enqueueing jobs
type RedisProducer struct {
	consumer *queue.RedisConsumer // Reuse Redis client
}

// NewStreamProcessServer creates a new gRPC server
func NewStreamProcessServer(consumer *queue.RedisConsumer) *StreamProcessServer {
	metrics := &ServerMetrics{
		RequestsTotal: prometheus.NewCounterVec(
			prometheus.CounterOpts{
				Name: "streamprocess_grpc_requests_total",
				Help: "Total number of gRPC requests",
			},
			[]string{"service", "method", "status"},
		),
		RequestDuration: prometheus.NewHistogramVec(
			prometheus.HistogramOpts{
				Name:    "streamprocess_grpc_request_duration_seconds",
				Help:    "gRPC request duration",
				Buckets: prometheus.ExponentialBuckets(0.001, 2, 15),
			},
			[]string{"service", "method"},
		),
		ActiveStreams: prometheus.NewGauge(
			prometheus.GaugeOpts{
				Name: "streamprocess_active_streams",
				Help: "Number of active streaming connections",
			},
		),
		QueuedJobs: prometheus.NewGaugeVec(
			prometheus.GaugeOpts{
				Name: "streamprocess_queued_jobs",
				Help: "Number of queued jobs by type",
			},
			[]string{"job_type"},
		),
		ProcessingErrors: prometheus.NewCounterVec(
			prometheus.CounterOpts{
				Name: "streamprocess_processing_errors_total",
				Help: "Total processing errors",
			},
			[]string{"service", "error_type"},
		),
	}

	// Register metrics
	prometheus.MustRegister(
		metrics.RequestsTotal,
		metrics.RequestDuration,
		metrics.ActiveStreams,
		metrics.QueuedJobs,
		metrics.ProcessingErrors,
	)

	return &StreamProcessServer{
		consumer: consumer,
		producer: &RedisProducer{consumer: consumer},
		logger:   logrus.New(),
		metrics:  metrics,
	}
}

// StreamingRecognize handles streaming STT requests
func (s *StreamProcessServer) StreamingRecognize(stream pb.STTService_StreamingRecognizeServer) error {
	s.metrics.ActiveStreams.Inc()
	defer s.metrics.ActiveStreams.Dec()

	startTime := time.Now()
	defer func() {
		s.metrics.RequestDuration.WithLabelValues("STTService", "StreamingRecognize").Observe(time.Since(startTime).Seconds())
	}()

	s.logger.Info("New streaming STT connection established")

	ctx := stream.Context()
	sessionID := fmt.Sprintf("stream_%d", time.Now().UnixNano())
	
	// Audio buffer for accumulating chunks
	var audioBuffer [][]byte
	var config *pb.StreamingRecognitionConfig

	for {
		req, err := stream.Recv()
		if err == io.EOF {
			// Client closed stream
			s.logger.WithField("session_id", sessionID).Info("Streaming STT connection closed")
			s.metrics.RequestsTotal.WithLabelValues("STTService", "StreamingRecognize", "success").Inc()
			return nil
		}
		if err != nil {
			s.logger.WithError(err).WithField("session_id", sessionID).Error("Error receiving stream")
			s.metrics.RequestsTotal.WithLabelValues("STTService", "StreamingRecognize", "error").Inc()
			s.metrics.ProcessingErrors.WithLabelValues("STTService", "stream_receive").Inc()
			return err
		}

		switch x := req.Request.(type) {
		case *pb.StreamingRecognizeRequest_Config:
			// Initial configuration
			config = x.Config
			s.logger.WithFields(logrus.Fields{
				"session_id": sessionID,
				"language":   config.AudioConfig.LanguageCode,
				"model":      config.RecognitionConfig.Model,
			}).Info("Received streaming configuration")

		case *pb.StreamingRecognizeRequest_AudioChunk:
			// Audio chunk received
			chunk := x.AudioChunk
			audioBuffer = append(audioBuffer, chunk.Content)

			// Process accumulated audio if we have enough or if this is final
			if len(audioBuffer) >= 10 || chunk.IsFinal { // Process every ~1 second or on final
				if err := s.processStreamingAudio(ctx, stream, sessionID, audioBuffer, config, chunk.IsFinal); err != nil {
					s.logger.WithError(err).WithField("session_id", sessionID).Error("Error processing streaming audio")
					s.metrics.ProcessingErrors.WithLabelValues("STTService", "audio_processing").Inc()
					return err
				}
				
				// Reset buffer if not final
				if !chunk.IsFinal {
					audioBuffer = audioBuffer[:0]
				}
			}
		}
	}
}

// processStreamingAudio processes accumulated audio chunks
func (s *StreamProcessServer) processStreamingAudio(ctx context.Context, stream pb.STTService_StreamingRecognizeServer, sessionID string, audioBuffer [][]byte, config *pb.StreamingRecognitionConfig, isFinal bool) error {
	// Combine audio chunks
	var combinedAudio []byte
	for _, chunk := range audioBuffer {
		combinedAudio = append(combinedAudio, chunk...)
	}

	// Create job for processing
	jobData := map[string]interface{}{
		"session_id":     sessionID,
		"audio_content":  combinedAudio,
		"language":       config.AudioConfig.LanguageCode,
		"model":          config.RecognitionConfig.Model,
		"is_final":       isFinal,
		"enable_partial": config.EnablePartialResults,
	}

	// Enqueue for processing with high priority for streaming
	jobID, err := s.producer.EnqueueSTTJob(jobData, 2) // HIGH priority
	if err != nil {
		return fmt.Errorf("failed to enqueue streaming STT job: %w", err)
	}

	s.metrics.QueuedJobs.WithLabelValues("stt_streaming").Inc()

	// Wait for result with short timeout for partial results
	timeout := 500 * time.Millisecond
	if isFinal {
		timeout = 5 * time.Second
	}

	result, err := s.waitForResult(ctx, jobID, timeout)
	if err != nil {
		// For partial results, send intermediate response
		if !isFinal {
			return stream.Send(&pb.StreamingRecognizeResponse{
				Status: pb.STATUS_PARTIAL,
				Result: &pb.RecognitionResult{
					Alternatives: []*pb.SpeechRecognitionAlternative{
						{
							Transcript: "",
							Confidence: 0.0,
						},
					},
					IsFinal: false,
				},
			})
		}
		return fmt.Errorf("failed to get processing result: %w", err)
	}

	// Parse result and send response
	var sttResult map[string]interface{}
	if err := json.Unmarshal(result, &sttResult); err != nil {
		return fmt.Errorf("failed to parse STT result: %w", err)
	}

	transcription, _ := sttResult["transcription"].(string)
	confidence, _ := sttResult["confidence"].(float64)

	return stream.Send(&pb.StreamingRecognizeResponse{
		Status: pb.STATUS_COMPLETED,
		Result: &pb.RecognitionResult{
			Alternatives: []*pb.SpeechRecognitionAlternative{
				{
					Transcript: transcription,
					Confidence: float32(confidence),
				},
			},
			IsFinal: isFinal,
		},
	})
}

// Recognize handles single STT requests
func (s *StreamProcessServer) Recognize(ctx context.Context, req *pb.RecognizeRequest) (*pb.RecognizeResponse, error) {
	startTime := time.Now()
	defer func() {
		s.metrics.RequestDuration.WithLabelValues("STTService", "Recognize").Observe(time.Since(startTime).Seconds())
	}()

	// Create job for processing
	jobData := map[string]interface{}{
		"audio_content": req.AudioContent,
		"language":      req.AudioConfig.LanguageCode,
		"model":         req.RecognitionConfig.Model,
		"beam_size":     req.RecognitionConfig.BeamSize,
	}

	// Enqueue for processing
	jobID, err := s.producer.EnqueueSTTJob(jobData, 1) // NORMAL priority
	if err != nil {
		s.metrics.RequestsTotal.WithLabelValues("STTService", "Recognize", "error").Inc()
		s.metrics.ProcessingErrors.WithLabelValues("STTService", "job_enqueue").Inc()
		return nil, fmt.Errorf("failed to enqueue STT job: %w", err)
	}

	s.metrics.QueuedJobs.WithLabelValues("stt_recognize").Inc()

	// Wait for result
	result, err := s.waitForResult(ctx, jobID, 30*time.Second)
	if err != nil {
		s.metrics.RequestsTotal.WithLabelValues("STTService", "Recognize", "error").Inc()
		s.metrics.ProcessingErrors.WithLabelValues("STTService", "result_wait").Inc()
		return nil, fmt.Errorf("failed to get processing result: %w", err)
	}

	// Parse result
	var sttResult map[string]interface{}
	if err := json.Unmarshal(result, &sttResult); err != nil {
		s.metrics.RequestsTotal.WithLabelValues("STTService", "Recognize", "error").Inc()
		return nil, fmt.Errorf("failed to parse STT result: %w", err)
	}

	transcription, _ := sttResult["transcription"].(string)
	confidence, _ := sttResult["confidence"].(float64)
	language, _ := sttResult["language"].(string)

	s.metrics.RequestsTotal.WithLabelValues("STTService", "Recognize", "success").Inc()

	return &pb.RecognizeResponse{
		Status: pb.STATUS_COMPLETED,
		Result: &pb.RecognitionResult{
			Alternatives: []*pb.SpeechRecognitionAlternative{
				{
					Transcript: transcription,
					Confidence: float32(confidence),
				},
			},
			IsFinal:  true,
			Language: language,
		},
	}, nil
}

// ProcessDocument handles OCR requests
func (s *StreamProcessServer) ProcessDocument(ctx context.Context, req *pb.DocumentRequest) (*pb.OCRResponse, error) {
	startTime := time.Now()
	defer func() {
		s.metrics.RequestDuration.WithLabelValues("OCRService", "ProcessDocument").Observe(time.Since(startTime).Seconds())
	}()

	// Create job for processing
	jobData := map[string]interface{}{
		"image_content": req.ImageContent,
		"languages":     req.Config.Languages,
		"detect_layout": req.Config.DetectLayout,
		"engine":        "paddleocr", // Default engine
	}

	// Enqueue for processing
	jobID, err := s.producer.EnqueueOCRJob(jobData, 1) // NORMAL priority
	if err != nil {
		s.metrics.RequestsTotal.WithLabelValues("OCRService", "ProcessDocument", "error").Inc()
		s.metrics.ProcessingErrors.WithLabelValues("OCRService", "job_enqueue").Inc()
		return nil, fmt.Errorf("failed to enqueue OCR job: %w", err)
	}

	s.metrics.QueuedJobs.WithLabelValues("ocr_process").Inc()

	// Wait for result
	result, err := s.waitForResult(ctx, jobID, 60*time.Second) // OCR can take longer
	if err != nil {
		s.metrics.RequestsTotal.WithLabelValues("OCRService", "ProcessDocument", "error").Inc()
		s.metrics.ProcessingErrors.WithLabelValues("OCRService", "result_wait").Inc()
		return nil, fmt.Errorf("failed to get processing result: %w", err)
	}

	// Parse result
	var ocrResult map[string]interface{}
	if err := json.Unmarshal(result, &ocrResult); err != nil {
		s.metrics.RequestsTotal.WithLabelValues("OCRService", "ProcessDocument", "error").Inc()
		return nil, fmt.Errorf("failed to parse OCR result: %w", err)
	}

	text, _ := ocrResult["text"].(string)
	confidence, _ := ocrResult["confidence"].(float64)

	s.metrics.RequestsTotal.WithLabelValues("OCRService", "ProcessDocument", "success").Inc()

	return &pb.OCRResponse{
		Status: pb.STATUS_COMPLETED,
		Text:   text,
		Confidence: float32(confidence),
	}, nil
}

// waitForResult waits for job processing result
func (s *StreamProcessServer) waitForResult(ctx context.Context, jobID string, timeout time.Duration) ([]byte, error) {
	ctx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()

	ticker := time.NewTicker(100 * time.Millisecond)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		case <-ticker.C:
			// Check for result (this would integrate with Redis result storage)
			// For now, simulate result retrieval
			// In real implementation, this would call consumer.GetResult(jobID)
			
			// Simulate processing delay
			time.Sleep(50 * time.Millisecond)
			
			// Return mock result for demo
			return []byte(`{"transcription":"Hello world","confidence":0.95,"language":"en"}`), nil
		}
	}
}

// EnqueueSTTJob enqueues an STT job
func (p *RedisProducer) EnqueueSTTJob(data map[string]interface{}, priority int) (string, error) {
	job := &queue.Job{
		Type:     "stt_transcribe",
		Priority: priority,
		Data:     data,
	}
	
	// This would integrate with the Redis producer
	// For now, return a mock job ID
	return fmt.Sprintf("stt_%d", time.Now().UnixNano()), nil
}

// EnqueueOCRJob enqueues an OCR job
func (p *RedisProducer) EnqueueOCRJob(data map[string]interface{}, priority int) (string, error) {
	job := &queue.Job{
		Type:     "ocr_process",
		Priority: priority,
		Data:     data,
	}
	
	// This would integrate with the Redis producer
	// For now, return a mock job ID
	return fmt.Sprintf("ocr_%d", time.Now().UnixNano()), nil
}

// Config represents server configuration
type Config struct {
	GRPCPort      int    `mapstructure:"grpc_port"`
	HTTPPort      int    `mapstructure:"http_port"`
	MetricsPort   int    `mapstructure:"metrics_port"`
	RedisAddr     string `mapstructure:"redis_addr"`
	RedisPassword string `mapstructure:"redis_password"`
	RedisDB       int    `mapstructure:"redis_db"`
	LogLevel      string `mapstructure:"log_level"`
}

var rootCmd = &cobra.Command{
	Use:   "grpc-server",
	Short: "StreamProcess gRPC Streaming Server in Go",
	Long:  "High-performance gRPC streaming server for STT and OCR services",
	Run: func(cmd *cobra.Command, args []string) {
		// Load configuration
		var config Config
		if err := viper.Unmarshal(&config); err != nil {
			logrus.Fatalf("Failed to unmarshal config: %v", err)
		}

		// Set log level
		level, err := logrus.ParseLevel(config.LogLevel)
		if err != nil {
			level = logrus.InfoLevel
		}
		logrus.SetLevel(level)

		// Create Redis consumer for job management
		consumer, err := queue.NewRedisConsumer(
			config.RedisAddr,
			config.RedisPassword,
			config.RedisDB,
			"grpc_server",
			"grpc_server_1",
		)
		if err != nil {
			logrus.Fatalf("Failed to create Redis consumer: %v", err)
		}
		defer consumer.Close()

		// Create gRPC server
		server := NewStreamProcessServer(consumer)

		// Start gRPC server
		lis, err := net.Listen("tcp", fmt.Sprintf(":%d", config.GRPCPort))
		if err != nil {
			logrus.Fatalf("Failed to listen on port %d: %v", config.GRPCPort, err)
		}

		grpcServer := grpc.NewServer()
		pb.RegisterSTTServiceServer(grpcServer, server)
		pb.RegisterOCRServiceServer(grpcServer, server)
		reflection.Register(grpcServer)

		// Start HTTP gateway
		go func() {
			ctx := context.Background()
			ctx, cancel := context.WithCancel(ctx)
			defer cancel()

			mux := runtime.NewServeMux()
			opts := []grpc.DialOption{grpc.WithTransportCredentials(insecure.NewCredentials())}
			
			if err := pb.RegisterSTTServiceHandlerFromEndpoint(ctx, mux, fmt.Sprintf("localhost:%d", config.GRPCPort), opts); err != nil {
				logrus.WithError(err).Error("Failed to register STT service handler")
				return
			}
			
			if err := pb.RegisterOCRServiceHandlerFromEndpoint(ctx, mux, fmt.Sprintf("localhost:%d", config.GRPCPort), opts); err != nil {
				logrus.WithError(err).Error("Failed to register OCR service handler")
				return
			}

			logrus.Infof("Starting HTTP gateway on port %d", config.HTTPPort)
			if err := http.ListenAndServe(fmt.Sprintf(":%d", config.HTTPPort), mux); err != nil {
				logrus.WithError(err).Error("HTTP gateway failed")
			}
		}()

		// Start metrics server
		go func() {
			http.Handle("/metrics", promhttp.Handler())
			http.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) {
				w.WriteHeader(http.StatusOK)
				w.Write([]byte("OK"))
			})
			
			logrus.Infof("Starting metrics server on port %d", config.MetricsPort)
			if err := http.ListenAndServe(fmt.Sprintf(":%d", config.MetricsPort), nil); err != nil {
				logrus.WithError(err).Error("Metrics server failed")
			}
		}()

		// Handle graceful shutdown
		go func() {
			sigChan := make(chan os.Signal, 1)
			signal.Notify(sigChan, syscall.SIGINT, syscall.SIGTERM)
			<-sigChan
			logrus.Info("Received shutdown signal")
			grpcServer.GracefulStop()
		}()

		// Start gRPC server
		logrus.Infof("Starting gRPC server on port %d", config.GRPCPort)
		if err := grpcServer.Serve(lis); err != nil {
			logrus.Fatalf("gRPC server failed: %v", err)
		}

		logrus.Info("gRPC server stopped")
	},
}

func init() {
	// Set default configuration
	viper.SetDefault("grpc_port", 50051)
	viper.SetDefault("http_port", 8080)
	viper.SetDefault("metrics_port", 9092)
	viper.SetDefault("redis_addr", "localhost:6379")
	viper.SetDefault("redis_password", "")
	viper.SetDefault("redis_db", 0)
	viper.SetDefault("log_level", "info")

	// Bind environment variables
	viper.AutomaticEnv()
	viper.SetEnvPrefix("GRPC")

	// Configuration file
	viper.SetConfigName("config")
	viper.SetConfigType("yaml")
	viper.AddConfigPath(".")
	viper.AddConfigPath("/etc/streamprocess/")
	viper.ReadInConfig()
}

func main() {
	if err := rootCmd.Execute(); err != nil {
		logrus.Fatal(err)
	}
}