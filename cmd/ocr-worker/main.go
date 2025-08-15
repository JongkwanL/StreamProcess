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

	"github.com/otiai10/gosseract/v2"
	"github.com/prometheus/client_golang/prometheus/promhttp"
	"github.com/sirupsen/logrus"
	"github.com/spf13/cobra"
	"github.com/spf13/viper"
	"streamprocess/pkg/queue"
)

// OCRWorker handles Optical Character Recognition jobs
type OCRWorker struct {
	consumer      *queue.RedisConsumer
	pythonAPIURL  string
	tesseractPath string
	logger        *logrus.Logger
	httpClient    *http.Client
}

// OCRJobData represents OCR job input data
type OCRJobData struct {
	ImageURL     string                 `json:"image_url"`
	ImageContent []byte                 `json:"image_content,omitempty"`
	Languages    []string               `json:"languages"`
	Engine       string                 `json:"engine"` // "paddleocr" or "tesseract"
	Config       map[string]interface{} `json:"config"`
}

// OCRResult represents OCR processing result
type OCRResult struct {
	Text         string                 `json:"text"`
	Languages    []string               `json:"languages"`
	Confidence   float64               `json:"confidence"`
	Blocks       []OCRBlock            `json:"blocks"`
	Layout       *LayoutInfo           `json:"layout,omitempty"`
	ProcessedAt  time.Time             `json:"processed_at"`
	Duration     float64               `json:"duration"`
	Engine       string                `json:"engine"`
	Metadata     map[string]interface{} `json:"metadata"`
}

// OCRBlock represents a block of recognized text
type OCRBlock struct {
	Text       string      `json:"text"`
	BoundingBox BoundingBox `json:"bounding_box"`
	Confidence float64     `json:"confidence"`
	Language   string      `json:"language,omitempty"`
}

// BoundingBox represents text bounding box coordinates
type BoundingBox struct {
	X      int `json:"x"`
	Y      int `json:"y"`
	Width  int `json:"width"`
	Height int `json:"height"`
}

// LayoutInfo represents document layout information
type LayoutInfo struct {
	Type     string                 `json:"type"` // "document", "table", "form"
	Elements []LayoutElement        `json:"elements"`
	Metadata map[string]interface{} `json:"metadata"`
}

// LayoutElement represents a layout element
type LayoutElement struct {
	Type        string      `json:"type"` // "paragraph", "heading", "table", "image"
	BoundingBox BoundingBox `json:"bounding_box"`
	Content     string      `json:"content"`
}

// NewOCRWorker creates a new OCR worker
func NewOCRWorker(consumer *queue.RedisConsumer, pythonAPIURL, tesseractPath string) *OCRWorker {
	return &OCRWorker{
		consumer:      consumer,
		pythonAPIURL:  pythonAPIURL,
		tesseractPath: tesseractPath,
		logger:        logrus.New(),
		httpClient: &http.Client{
			Timeout: 60 * time.Second, // OCR can take longer
		},
	}
}

// ProcessJob implements the JobProcessor interface
func (w *OCRWorker) ProcessJob(ctx context.Context, job *queue.Job) (interface{}, error) {
	w.logger.WithFields(logrus.Fields{
		"job_id":   job.ID,
		"job_type": job.Type,
	}).Info("Processing OCR job")

	start := time.Now()

	// Parse OCR job data
	var ocrData OCRJobData
	jobDataBytes, err := json.Marshal(job.Data)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal job data: %w", err)
	}

	if err := json.Unmarshal(jobDataBytes, &ocrData); err != nil {
		return nil, fmt.Errorf("failed to parse OCR job data: %w", err)
	}

	// Validate required fields
	if ocrData.ImageURL == "" && len(ocrData.ImageContent) == 0 {
		return nil, fmt.Errorf("either image_url or image_content must be provided")
	}

	// Set default engine
	if ocrData.Engine == "" {
		ocrData.Engine = "paddleocr"
	}

	// Process image based on engine and job type
	var result *OCRResult
	switch ocrData.Engine {
	case "tesseract":
		result, err = w.processWithTesseract(ctx, &ocrData)
	case "paddleocr":
		result, err = w.processWithPaddleOCR(ctx, &ocrData)
	default:
		return nil, fmt.Errorf("unknown OCR engine: %s", ocrData.Engine)
	}

	if err != nil {
		return nil, fmt.Errorf("OCR processing failed: %w", err)
	}

	// Add processing metadata
	result.ProcessedAt = time.Now()
	result.Duration = time.Since(start).Seconds()
	result.Engine = ocrData.Engine

	w.logger.WithFields(logrus.Fields{
		"job_id":     job.ID,
		"text_length": len(result.Text),
		"blocks":     len(result.Blocks),
		"confidence": result.Confidence,
		"duration":   result.Duration,
		"engine":     result.Engine,
	}).Info("OCR job completed successfully")

	return result, nil
}

// processWithTesseract processes image using Tesseract OCR
func (w *OCRWorker) processWithTesseract(ctx context.Context, data *OCRJobData) (*OCRResult, error) {
	client := gosseract.NewClient()
	defer client.Close()

	// Set languages
	if len(data.Languages) > 0 {
		langString := ""
		for i, lang := range data.Languages {
			if i > 0 {
				langString += "+"
			}
			langString += lang
		}
		client.SetLanguage(langString)
	}

	// Set image source
	if data.ImageURL != "" {
		// Download image
		resp, err := w.httpClient.Get(data.ImageURL)
		if err != nil {
			return nil, fmt.Errorf("failed to download image: %w", err)
		}
		defer resp.Body.Close()

		imageData, err := io.ReadAll(resp.Body)
		if err != nil {
			return nil, fmt.Errorf("failed to read image data: %w", err)
		}
		client.SetImageFromBytes(imageData)
	} else {
		client.SetImageFromBytes(data.ImageContent)
	}

	// Get text
	text, err := client.Text()
	if err != nil {
		return nil, fmt.Errorf("tesseract text extraction failed: %w", err)
	}

	// Get bounding boxes (requires additional API calls)
	boxes, err := client.GetBoundingBoxes(gosseract.RIL_WORD)
	if err != nil {
		w.logger.WithError(err).Warn("Failed to get bounding boxes")
		boxes = []gosseract.BoundingBox{}
	}

	// Convert to our format
	var ocrBlocks []OCRBlock
	for _, box := range boxes {
		ocrBlocks = append(ocrBlocks, OCRBlock{
			Text: box.Word,
			BoundingBox: BoundingBox{
				X:      box.Box.Min.X,
				Y:      box.Box.Min.Y,
				Width:  box.Box.Max.X - box.Box.Min.X,
				Height: box.Box.Max.Y - box.Box.Min.Y,
			},
			Confidence: float64(box.Confidence),
		})
	}

	// Calculate average confidence
	avgConfidence := 0.0
	if len(ocrBlocks) > 0 {
		totalConfidence := 0.0
		for _, block := range ocrBlocks {
			totalConfidence += block.Confidence
		}
		avgConfidence = totalConfidence / float64(len(ocrBlocks))
	}

	return &OCRResult{
		Text:       text,
		Languages:  data.Languages,
		Confidence: avgConfidence,
		Blocks:     ocrBlocks,
		Metadata: map[string]interface{}{
			"tesseract_version": gosseract.Version(),
		},
	}, nil
}

// processWithPaddleOCR processes image using PaddleOCR via Python service
func (w *OCRWorker) processWithPaddleOCR(ctx context.Context, data *OCRJobData) (*OCRResult, error) {
	// Prepare request to Python OCR service
	reqData := map[string]interface{}{
		"image_url":     data.ImageURL,
		"image_content": data.ImageContent,
		"languages":     data.Languages,
		"engine":        "paddleocr",
		"config":        data.Config,
	}

	reqBody, err := json.Marshal(reqData)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	// Call Python OCR service
	req, err := http.NewRequestWithContext(ctx, "POST", 
		fmt.Sprintf("%s/internal/ocr/process", w.pythonAPIURL), 
		io.NopCloser(io.Reader(nil)))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	req.Header.Set("Content-Type", "application/json")
	req.Body = io.NopCloser(io.Reader(reqBody))

	resp, err := w.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to call Python OCR service: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("Python OCR service returned status %d", resp.StatusCode)
	}

	// Parse response
	var result OCRResult
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("failed to decode OCR response: %w", err)
	}

	return &result, nil
}

// Config represents worker configuration
type Config struct {
	RedisAddr      string `mapstructure:"redis_addr"`
	RedisPassword  string `mapstructure:"redis_password"`
	RedisDB        int    `mapstructure:"redis_db"`
	ConsumerGroup  string `mapstructure:"consumer_group"`
	ConsumerName   string `mapstructure:"consumer_name"`
	PythonAPIURL   string `mapstructure:"python_api_url"`
	TesseractPath  string `mapstructure:"tesseract_path"`
	MetricsPort    int    `mapstructure:"metrics_port"`
	LogLevel       string `mapstructure:"log_level"`
}

var rootCmd = &cobra.Command{
	Use:   "ocr-worker",
	Short: "StreamProcess OCR Worker in Go",
	Long:  "High-performance OCR worker with Tesseract and PaddleOCR support",
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

		// Create OCR worker
		worker := NewOCRWorker(consumer, config.PythonAPIURL, config.TesseractPath)

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
		logrus.Info("Starting OCR worker")
		if err := consumer.ConsumeJobs(worker); err != nil && err != context.Canceled {
			logrus.Fatalf("Worker failed: %v", err)
		}

		logrus.Info("OCR worker stopped")
	},
}

func init() {
	// Set default configuration
	viper.SetDefault("redis_addr", "localhost:6379")
	viper.SetDefault("redis_password", "")
	viper.SetDefault("redis_db", 0)
	viper.SetDefault("consumer_group", "ocr_workers")
	viper.SetDefault("consumer_name", "ocr_worker_1")
	viper.SetDefault("python_api_url", "http://localhost:8000")
	viper.SetDefault("tesseract_path", "/usr/bin/tesseract")
	viper.SetDefault("metrics_port", 9091)
	viper.SetDefault("log_level", "info")

	// Bind environment variables
	viper.AutomaticEnv()
	viper.SetEnvPrefix("OCR")

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