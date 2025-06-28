package main

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/prometheus/client_golang/prometheus/promhttp"
	"github.com/sirupsen/logrus"
	"github.com/spf13/cobra"
	"github.com/spf13/viper"
	"github.com/streamprocess/streamprocess/pkg/queue"
)

// STTWorker handles Speech-to-Text processing jobs
type STTWorker struct {
	consumer     *queue.RedisConsumer
	pythonAPIURL string
	logger       *logrus.Logger
	httpClient   *http.Client
}

// STTJobData represents STT job input data
type STTJobData struct {
	AudioURL      string            `json:"audio_url"`
	AudioContent  []byte            `json:"audio_content,omitempty"`
	Language      string            `json:"language"`
	Model         string            `json:"model"`
	Config        map[string]interface{} `json:"config"`
}

// STTResult represents STT processing result
type STTResult struct {
	Transcription string                 `json:"transcription"`
	Language      string                 `json:"language"`
	Confidence    float64               `json:"confidence"`
	Segments      []TranscriptSegment   `json:"segments"`
	Duration      float64               `json:"duration"`
	ProcessedAt   time.Time             `json:"processed_at"`
	Metadata      map[string]interface{} `json:"metadata"`
}

// TranscriptSegment represents a segment of transcribed text
type TranscriptSegment struct {
	Text       string  `json:"text"`
	Start      float64 `json:"start"`
	End        float64 `json:"end"`
	Confidence float64 `json:"confidence"`
}

// NewSTTWorker creates a new STT worker
func NewSTTWorker(consumer *queue.RedisConsumer, pythonAPIURL string) *STTWorker {
	return &STTWorker{
		consumer:     consumer,
		pythonAPIURL: pythonAPIURL,
		logger:       logrus.New(),
		httpClient: &http.Client{
			Timeout: 30 * time.Second,
		},
	}
}

// ProcessJob implements the JobProcessor interface
func (w *STTWorker) ProcessJob(ctx context.Context, job *queue.Job) (interface{}, error) {
	w.logger.WithFields(logrus.Fields{
		"job_id":   job.ID,
		"job_type": job.Type,
	}).Info("Processing STT job")

	start := time.Now()

	// Parse STT job data
	var sttData STTJobData
	jobDataBytes, err := json.Marshal(job.Data)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal job data: %w", err)
	}

	if err := json.Unmarshal(jobDataBytes, &sttData); err != nil {
		return nil, fmt.Errorf("failed to parse STT job data: %w", err)
	}

	// Validate required fields
	if sttData.AudioURL == "" && len(sttData.AudioContent) == 0 {
		return nil, fmt.Errorf("either audio_url or audio_content must be provided")
	}

	// Process audio based on job type
	var result *STTResult
	switch job.Type {
	case "stt_transcribe":
		result, err = w.transcribeAudio(ctx, &sttData)
	case "stt_streaming":
		result, err = w.processStreamingAudio(ctx, &sttData)
	case "stt_batch":
		result, err = w.processBatchAudio(ctx, &sttData)
	default:
		return nil, fmt.Errorf("unknown STT job type: %s", job.Type)
	}

	if err != nil {
		return nil, fmt.Errorf("STT processing failed: %w", err)
	}

	// Add processing metadata
	result.ProcessedAt = time.Now()
	result.Duration = time.Since(start).Seconds()

	w.logger.WithFields(logrus.Fields{
		"job_id":        job.ID,
		"transcription": result.Transcription[:min(50, len(result.Transcription))], // First 50 chars
		"language":      result.Language,
		"confidence":    result.Confidence,
		"duration":      result.Duration,
	}).Info("STT job completed successfully")

	return result, nil
}

// transcribeAudio handles standard transcription requests
func (w *STTWorker) transcribeAudio(ctx context.Context, data *STTJobData) (*STTResult, error) {
	// Prepare request to Python STT service
	reqData := map[string]interface{}{
		"audio_url":     data.AudioURL,
		"audio_content": data.AudioContent,
		"language":      data.Language,
		"model":         data.Model,
		"config":        data.Config,
	}

	reqBody, err := json.Marshal(reqData)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	// Call Python STT service
	req, err := http.NewRequestWithContext(ctx, "POST", 
		fmt.Sprintf("%s/internal/stt/transcribe", w.pythonAPIURL), 
		io.NopCloser(io.Reader(nil)))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	req.Header.Set("Content-Type", "application/json")
	req.Body = io.NopCloser(io.Reader(reqBody))

	resp, err := w.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to call Python STT service: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("Python STT service returned status %d", resp.StatusCode)
	}

	// Parse response
	var result STTResult
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("failed to decode STT response: %w", err)
	}

	return &result, nil
}

// processStreamingAudio handles streaming transcription
func (w *STTWorker) processStreamingAudio(ctx context.Context, data *STTJobData) (*STTResult, error) {
	// For streaming, we accumulate results and return final transcription
	return w.transcribeAudio(ctx, data)
}

// processBatchAudio handles batch transcription
func (w *STTWorker) processBatchAudio(ctx context.Context, data *STTJobData) (*STTResult, error) {
	// Batch processing might include additional optimizations
	return w.transcribeAudio(ctx, data)
}

// Config represents worker configuration
type Config struct {
	RedisAddr      string `mapstructure:"redis_addr"`
	RedisPassword  string `mapstructure:"redis_password"`
	RedisDB        int    `mapstructure:"redis_db"`
	ConsumerGroup  string `mapstructure:"consumer_group"`
	ConsumerName   string `mapstructure:"consumer_name"`
	PythonAPIURL   string `mapstructure:"python_api_url"`
	MetricsPort    int    `mapstructure:"metrics_port"`
	LogLevel       string `mapstructure:"log_level"`
}

var rootCmd = &cobra.Command{
	Use:   "stt-worker",
	Short: "StreamProcess STT Worker in Go",
	Long:  "High-performance STT worker using Go goroutines for concurrent processing",
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

		// Create Redis consumer
		consumer, err := queue.NewRedisConsumer(
			config.RedisAddr,
			config.RedisPassword,
			config.RedisDB,
			config.ConsumerGroup,
			config.ConsumerName,
		)
		if err != nil {
			logrus.Fatalf("Failed to create Redis consumer: %v", err)
		}
		defer consumer.Close()

		// Create STT worker
		worker := NewSTTWorker(consumer, config.PythonAPIURL)

		// Start metrics server
		go func() {
			http.Handle("/metrics", promhttp.Handler())
			logrus.Infof("Starting metrics server on port %d", config.MetricsPort)
			if err := http.ListenAndServe(fmt.Sprintf(":%d", config.MetricsPort), nil); err != nil {
				logrus.WithError(err).Error("Metrics server failed")
			}
		}()

		// Start metrics updater
		go func() {
			ticker := time.NewTicker(10 * time.Second)
			defer ticker.Stop()
			
			for {
				select {
				case <-ticker.C:
					consumer.UpdateMetrics()
				case <-cmd.Context().Done():
					return
				}
			}
		}()

		// Handle graceful shutdown
		ctx, cancel := context.WithCancel(context.Background())
		go func() {
			sigChan := make(chan os.Signal, 1)
			signal.Notify(sigChan, syscall.SIGINT, syscall.SIGTERM)
			<-sigChan
			logrus.Info("Received shutdown signal")
			cancel()
		}()

		// Start consuming jobs
		logrus.Info("Starting STT worker")
		if err := consumer.ConsumeJobs(worker); err != nil && err != context.Canceled {
			logrus.Fatalf("Worker failed: %v", err)
		}

		logrus.Info("STT worker stopped")
	},
}

func init() {
	// Set default configuration
	viper.SetDefault("redis_addr", "localhost:6379")
	viper.SetDefault("redis_password", "")
	viper.SetDefault("redis_db", 0)
	viper.SetDefault("consumer_group", "stt_workers")
	viper.SetDefault("consumer_name", "stt_worker_1")
	viper.SetDefault("python_api_url", "http://localhost:8000")
	viper.SetDefault("metrics_port", 9090)
	viper.SetDefault("log_level", "info")

	// Bind environment variables
	viper.AutomaticEnv()
	viper.SetEnvPrefix("STT")

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

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}