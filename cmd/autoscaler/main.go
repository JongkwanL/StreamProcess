package main

import (
	"context"
	"fmt"
	"math"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promhttp"
	"github.com/sirupsen/logrus"
	"github.com/spf13/cobra"
	"github.com/spf13/viper"
	
	appsv1 "k8s.io/api/apps/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/rest"
	"k8s.io/client-go/tools/clientcmd"
	metricsv1beta1 "k8s.io/metrics/clientset/versioned"
	
	"github.com/go-redis/redis/v8"
	"net/http"
)

// AutoscalerController handles intelligent autoscaling for StreamProcess workers
type AutoscalerController struct {
	k8sClient     kubernetes.Interface
	metricsClient metricsv1beta1.Interface
	redisClient   *redis.Client
	logger        *logrus.Logger
	metrics       *ControllerMetrics
	config        *AutoscalerConfig
	
	// PID controller state
	pidControllers map[string]*PIDController
}

// AutoscalerConfig represents controller configuration
type AutoscalerConfig struct {
	Namespace         string        `mapstructure:"namespace"`
	RedisAddr         string        `mapstructure:"redis_addr"`
	RedisPassword     string        `mapstructure:"redis_password"`
	RedisDB           int           `mapstructure:"redis_db"`
	CheckInterval     time.Duration `mapstructure:"check_interval"`
	MetricsPort       int           `mapstructure:"metrics_port"`
	LogLevel          string        `mapstructure:"log_level"`
	
	// Scaling parameters
	STTWorkers AutoscalerParams `mapstructure:"stt_workers"`
	OCRWorkers AutoscalerParams `mapstructure:"ocr_workers"`
}

// AutoscalerParams represents scaling parameters for a worker type
type AutoscalerParams struct {
	DeploymentName   string  `mapstructure:"deployment_name"`
	MinReplicas      int32   `mapstructure:"min_replicas"`
	MaxReplicas      int32   `mapstructure:"max_replicas"`
	TargetQueueDepth int     `mapstructure:"target_queue_depth"`
	TargetCPU        float64 `mapstructure:"target_cpu"`
	TargetMemory     float64 `mapstructure:"target_memory"`
	
	// PID controller parameters
	Kp float64 `mapstructure:"kp"` // Proportional gain
	Ki float64 `mapstructure:"ki"` // Integral gain  
	Kd float64 `mapstructure:"kd"` // Derivative gain
}

// PIDController implements PID control algorithm
type PIDController struct {
	Kp, Ki, Kd       float64
	integral         float64
	previousError    float64
	previousTime     time.Time
	target           float64
}

// ControllerMetrics for Prometheus monitoring
type ControllerMetrics struct {
	ScalingDecisions    *prometheus.CounterVec
	CurrentReplicas     *prometheus.GaugeVec
	DesiredReplicas     *prometheus.GaugeVec
	QueueDepth          *prometheus.GaugeVec
	CPUUtilization      *prometheus.GaugeVec
	MemoryUtilization   *prometheus.GaugeVec
	ScalingLatency      *prometheus.HistogramVec
	PIDOutput           *prometheus.GaugeVec
}

// WorkerMetrics represents current worker metrics
type WorkerMetrics struct {
	CurrentReplicas int32
	QueueDepth      int
	CPUUtilization  float64
	MemoryUtil      float64
	AvgLatency      float64
}

// NewAutoscalerController creates a new autoscaler controller
func NewAutoscalerController(config *AutoscalerConfig) (*AutoscalerController, error) {
	// Create Kubernetes client
	var kubeConfig *rest.Config
	var err error
	
	if kubeConfigPath := os.Getenv("KUBECONFIG"); kubeConfigPath != "" {
		kubeConfig, err = clientcmd.BuildConfigFromFlags("", kubeConfigPath)
	} else {
		kubeConfig, err = rest.InClusterConfig()
	}
	if err != nil {
		return nil, fmt.Errorf("failed to create kube config: %w", err)
	}

	k8sClient, err := kubernetes.NewForConfig(kubeConfig)
	if err != nil {
		return nil, fmt.Errorf("failed to create k8s client: %w", err)
	}

	metricsClient, err := metricsv1beta1.NewForConfig(kubeConfig)
	if err != nil {
		return nil, fmt.Errorf("failed to create metrics client: %w", err)
	}

	// Create Redis client
	redisClient := redis.NewClient(&redis.Options{
		Addr:     config.RedisAddr,
		Password: config.RedisPassword,
		DB:       config.RedisDB,
	})

	// Test Redis connection
	ctx := context.Background()
	if err := redisClient.Ping(ctx).Err(); err != nil {
		return nil, fmt.Errorf("failed to connect to Redis: %w", err)
	}

	// Initialize metrics
	metrics := &ControllerMetrics{
		ScalingDecisions: prometheus.NewCounterVec(
			prometheus.CounterOpts{
				Name: "streamprocess_scaling_decisions_total",
				Help: "Total number of scaling decisions",
			},
			[]string{"worker_type", "direction"},
		),
		CurrentReplicas: prometheus.NewGaugeVec(
			prometheus.GaugeOpts{
				Name: "streamprocess_current_replicas",
				Help: "Current number of replicas",
			},
			[]string{"worker_type"},
		),
		DesiredReplicas: prometheus.NewGaugeVec(
			prometheus.GaugeOpts{
				Name: "streamprocess_desired_replicas",
				Help: "Desired number of replicas",
			},
			[]string{"worker_type"},
		),
		QueueDepth: prometheus.NewGaugeVec(
			prometheus.GaugeOpts{
				Name: "streamprocess_autoscaler_queue_depth",
				Help: "Current queue depth observed by autoscaler",
			},
			[]string{"worker_type"},
		),
		CPUUtilization: prometheus.NewGaugeVec(
			prometheus.GaugeOpts{
				Name: "streamprocess_autoscaler_cpu_utilization",
				Help: "CPU utilization observed by autoscaler",
			},
			[]string{"worker_type"},
		),
		MemoryUtilization: prometheus.NewGaugeVec(
			prometheus.GaugeOpts{
				Name: "streamprocess_autoscaler_memory_utilization",
				Help: "Memory utilization observed by autoscaler",
			},
			[]string{"worker_type"},
		),
		ScalingLatency: prometheus.NewHistogramVec(
			prometheus.HistogramOpts{
				Name:    "streamprocess_scaling_latency_seconds",
				Help:    "Time taken to complete scaling operation",
				Buckets: prometheus.ExponentialBuckets(0.1, 2, 10),
			},
			[]string{"worker_type"},
		),
		PIDOutput: prometheus.NewGaugeVec(
			prometheus.GaugeOpts{
				Name: "streamprocess_pid_output",
				Help: "PID controller output",
			},
			[]string{"worker_type"},
		),
	}

	// Register metrics
	prometheus.MustRegister(
		metrics.ScalingDecisions,
		metrics.CurrentReplicas,
		metrics.DesiredReplicas,
		metrics.QueueDepth,
		metrics.CPUUtilization,
		metrics.MemoryUtilization,
		metrics.ScalingLatency,
		metrics.PIDOutput,
	)

	// Initialize PID controllers
	pidControllers := map[string]*PIDController{
		"stt": NewPIDController(config.STTWorkers.Kp, config.STTWorkers.Ki, config.STTWorkers.Kd, float64(config.STTWorkers.TargetQueueDepth)),
		"ocr": NewPIDController(config.OCRWorkers.Kp, config.OCRWorkers.Ki, config.OCRWorkers.Kd, float64(config.OCRWorkers.TargetQueueDepth)),
	}

	return &AutoscalerController{
		k8sClient:      k8sClient,
		metricsClient:  metricsClient,
		redisClient:    redisClient,
		logger:         logrus.New(),
		metrics:        metrics,
		config:         config,
		pidControllers: pidControllers,
	}, nil
}

// NewPIDController creates a new PID controller
func NewPIDController(kp, ki, kd, target float64) *PIDController {
	return &PIDController{
		Kp:     kp,
		Ki:     ki,
		Kd:     kd,
		target: target,
	}
}

// Update calculates PID output based on current value
func (pid *PIDController) Update(currentValue float64) float64 {
	now := time.Now()
	
	if pid.previousTime.IsZero() {
		pid.previousTime = now
		pid.previousError = pid.target - currentValue
		return 0
	}

	dt := now.Sub(pid.previousTime).Seconds()
	error := pid.target - currentValue

	// Proportional term
	proportional := pid.Kp * error

	// Integral term
	pid.integral += error * dt
	integral := pid.Ki * pid.integral

	// Derivative term
	derivative := pid.Kd * (error - pid.previousError) / dt

	// Calculate output
	output := proportional + integral + derivative

	// Update for next iteration
	pid.previousError = error
	pid.previousTime = now

	return output
}

// Run starts the autoscaler control loop
func (c *AutoscalerController) Run(ctx context.Context) error {
	c.logger.Info("Starting autoscaler controller")

	ticker := time.NewTicker(c.config.CheckInterval)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-ticker.C:
			if err := c.runScalingLoop(ctx); err != nil {
				c.logger.WithError(err).Error("Scaling loop failed")
			}
		}
	}
}

// runScalingLoop executes one iteration of the scaling logic
func (c *AutoscalerController) runScalingLoop(ctx context.Context) error {
	// Scale STT workers
	if err := c.scaleWorkers(ctx, "stt", c.config.STTWorkers); err != nil {
		c.logger.WithError(err).Error("Failed to scale STT workers")
	}

	// Scale OCR workers
	if err := c.scaleWorkers(ctx, "ocr", c.config.OCRWorkers); err != nil {
		c.logger.WithError(err).Error("Failed to scale OCR workers")
	}

	return nil
}

// scaleWorkers handles scaling logic for a specific worker type
func (c *AutoscalerController) scaleWorkers(ctx context.Context, workerType string, params AutoscalerParams) error {
	start := time.Now()
	defer func() {
		c.metrics.ScalingLatency.WithLabelValues(workerType).Observe(time.Since(start).Seconds())
	}()

	// Get current metrics
	metrics, err := c.getWorkerMetrics(ctx, workerType, params.DeploymentName)
	if err != nil {
		return fmt.Errorf("failed to get worker metrics: %w", err)
	}

	// Update Prometheus metrics
	c.metrics.CurrentReplicas.WithLabelValues(workerType).Set(float64(metrics.CurrentReplicas))
	c.metrics.QueueDepth.WithLabelValues(workerType).Set(float64(metrics.QueueDepth))
	c.metrics.CPUUtilization.WithLabelValues(workerType).Set(metrics.CPUUtilization)
	c.metrics.MemoryUtilization.WithLabelValues(workerType).Set(metrics.MemoryUtil)

	// Calculate desired replicas using PID controller
	pid := c.pidControllers[workerType]
	pidOutput := pid.Update(float64(metrics.QueueDepth))
	c.metrics.PIDOutput.WithLabelValues(workerType).Set(pidOutput)

	// Calculate base replicas needed based on queue depth
	baseReplicas := c.calculateBaseReplicas(metrics, params)
	
	// Apply PID adjustment
	adjustment := pidOutput / 10.0 // Scale down PID output
	desiredReplicas := int32(math.Max(float64(baseReplicas) + adjustment, float64(params.MinReplicas)))
	
	// Apply resource utilization constraints
	desiredReplicas = c.applyResourceConstraints(desiredReplicas, metrics, params)
	
	// Clamp to min/max
	if desiredReplicas < params.MinReplicas {
		desiredReplicas = params.MinReplicas
	}
	if desiredReplicas > params.MaxReplicas {
		desiredReplicas = params.MaxReplicas
	}

	c.metrics.DesiredReplicas.WithLabelValues(workerType).Set(float64(desiredReplicas))

	// Only scale if there's a meaningful difference
	if desiredReplicas != metrics.CurrentReplicas {
		c.logger.WithFields(logrus.Fields{
			"worker_type":       workerType,
			"current_replicas":  metrics.CurrentReplicas,
			"desired_replicas":  desiredReplicas,
			"queue_depth":       metrics.QueueDepth,
			"cpu_utilization":   metrics.CPUUtilization,
			"memory_util":       metrics.MemoryUtil,
			"pid_output":        pidOutput,
		}).Info("Scaling workers")

		if err := c.scaleDeployment(ctx, params.DeploymentName, desiredReplicas); err != nil {
			return fmt.Errorf("failed to scale deployment: %w", err)
		}

		// Record scaling decision
		direction := "up"
		if desiredReplicas < metrics.CurrentReplicas {
			direction = "down"
		}
		c.metrics.ScalingDecisions.WithLabelValues(workerType, direction).Inc()
	}

	return nil
}

// calculateBaseReplicas calculates base replicas needed based on queue depth
func (c *AutoscalerController) calculateBaseReplicas(metrics *WorkerMetrics, params AutoscalerParams) int32 {
	if metrics.QueueDepth == 0 {
		return params.MinReplicas
	}

	// Calculate replicas needed to handle current queue depth
	// Assume each worker can handle target_queue_depth jobs efficiently
	baseReplicas := int32(math.Ceil(float64(metrics.QueueDepth) / float64(params.TargetQueueDepth)))
	
	// Add some buffer for incoming jobs
	bufferReplicas := int32(math.Ceil(float64(baseReplicas) * 0.2))
	
	return baseReplicas + bufferReplicas
}

// applyResourceConstraints adjusts replicas based on resource utilization
func (c *AutoscalerController) applyResourceConstraints(replicas int32, metrics *WorkerMetrics, params AutoscalerParams) int32 {
	// If CPU or memory is high, scale up more aggressively
	if metrics.CPUUtilization > params.TargetCPU || metrics.MemoryUtil > params.TargetMemory {
		resourcePressure := math.Max(metrics.CPUUtilization/params.TargetCPU, metrics.MemoryUtil/params.TargetMemory)
		adjustment := int32(math.Ceil(float64(replicas) * (resourcePressure - 1.0) * 0.5))
		replicas += adjustment
	}

	return replicas
}

// getWorkerMetrics collects current metrics for worker type
func (c *AutoscalerController) getWorkerMetrics(ctx context.Context, workerType, deploymentName string) (*WorkerMetrics, error) {
	// Get deployment info
	deployment, err := c.k8sClient.AppsV1().Deployments(c.config.Namespace).Get(ctx, deploymentName, metav1.GetOptions{})
	if err != nil {
		return nil, fmt.Errorf("failed to get deployment: %w", err)
	}

	currentReplicas := *deployment.Spec.Replicas

	// Get queue depth from Redis
	queueDepth, err := c.getQueueDepth(ctx, workerType)
	if err != nil {
		c.logger.WithError(err).Warn("Failed to get queue depth, using 0")
		queueDepth = 0
	}

	// Get resource utilization from metrics API
	cpuUtil, memUtil, err := c.getResourceUtilization(ctx, deploymentName)
	if err != nil {
		c.logger.WithError(err).Warn("Failed to get resource utilization, using defaults")
		cpuUtil, memUtil = 0.5, 0.5 // Default values
	}

	return &WorkerMetrics{
		CurrentReplicas: currentReplicas,
		QueueDepth:      queueDepth,
		CPUUtilization:  cpuUtil,
		MemoryUtil:      memUtil,
	}, nil
}

// getQueueDepth gets current queue depth from Redis
func (c *AutoscalerController) getQueueDepth(ctx context.Context, workerType string) (int, error) {
	// Queue keys based on worker type and priority
	var queueKeys []string
	switch workerType {
	case "stt":
		queueKeys = []string{
			"streamprocess:queue:low",
			"streamprocess:queue:normal", 
			"streamprocess:queue:high",
			"streamprocess:queue:realtime",
		}
	case "ocr":
		queueKeys = []string{
			"streamprocess:queue:low",
			"streamprocess:queue:normal",
			"streamprocess:queue:high", 
			"streamprocess:queue:realtime",
		}
	}

	totalDepth := 0
	for _, key := range queueKeys {
		length := c.redisClient.XLen(ctx, key).Val()
		totalDepth += int(length)
	}

	return totalDepth, nil
}

// getResourceUtilization gets CPU and memory utilization from metrics API
func (c *AutoscalerController) getResourceUtilization(ctx context.Context, deploymentName string) (float64, float64, error) {
	// Get pod metrics
	podMetrics, err := c.metricsClient.MetricsV1beta1().PodMetricses(c.config.Namespace).List(ctx, metav1.ListOptions{
		LabelSelector: fmt.Sprintf("app=%s", deploymentName),
	})
	if err != nil {
		return 0, 0, fmt.Errorf("failed to get pod metrics: %w", err)
	}

	if len(podMetrics.Items) == 0 {
		return 0, 0, fmt.Errorf("no pod metrics found")
	}

	// Calculate average utilization across all pods
	var totalCPU, totalMemory float64
	podCount := float64(len(podMetrics.Items))

	for _, pod := range podMetrics.Items {
		for _, container := range pod.Containers {
			// Convert CPU from millicores to cores
			cpuQuantity := container.Usage["cpu"]
			cpuCores := float64(cpuQuantity.MilliValue()) / 1000.0
			
			// Convert memory from bytes to percentage (assuming 8GB pods)
			memQuantity := container.Usage["memory"]
			memBytes := float64(memQuantity.Value())
			memPercent := memBytes / (8 * 1024 * 1024 * 1024) // 8GB in bytes

			totalCPU += cpuCores / 4.0 // Assuming 4 CPU limit
			totalMemory += memPercent
		}
	}

	avgCPU := totalCPU / podCount
	avgMemory := totalMemory / podCount

	return avgCPU, avgMemory, nil
}

// scaleDeployment scales a deployment to the desired replica count
func (c *AutoscalerController) scaleDeployment(ctx context.Context, deploymentName string, replicas int32) error {
	deployment, err := c.k8sClient.AppsV1().Deployments(c.config.Namespace).Get(ctx, deploymentName, metav1.GetOptions{})
	if err != nil {
		return fmt.Errorf("failed to get deployment: %w", err)
	}

	deployment.Spec.Replicas = &replicas

	_, err = c.k8sClient.AppsV1().Deployments(c.config.Namespace).Update(ctx, deployment, metav1.UpdateOptions{})
	if err != nil {
		return fmt.Errorf("failed to update deployment: %w", err)
	}

	c.logger.WithFields(logrus.Fields{
		"deployment": deploymentName,
		"replicas":   replicas,
	}).Info("Scaled deployment")

	return nil
}

// Config represents controller configuration
type Config struct {
	Namespace     string `mapstructure:"namespace"`
	RedisAddr     string `mapstructure:"redis_addr"`
	RedisPassword string `mapstructure:"redis_password"`
	RedisDB       int    `mapstructure:"redis_db"`
	CheckInterval string `mapstructure:"check_interval"`
	MetricsPort   int    `mapstructure:"metrics_port"`
	LogLevel      string `mapstructure:"log_level"`
	
	STTWorkers AutoscalerParams `mapstructure:"stt_workers"`
	OCRWorkers AutoscalerParams `mapstructure:"ocr_workers"`
}

var rootCmd = &cobra.Command{
	Use:   "autoscaler",
	Short: "StreamProcess Autoscaler Controller in Go",
	Long:  "Intelligent autoscaler using PID control and queue depth metrics",
	Run: func(cmd *cobra.Command, args []string) {
		// Load configuration
		var config Config
		if err := viper.Unmarshal(&config); err != nil {
			logrus.Fatalf("Failed to unmarshal config: %v", err)
		}

		// Parse check interval
		checkInterval, err := time.ParseDuration(config.CheckInterval)
		if err != nil {
			logrus.Fatalf("Failed to parse check interval: %v", err)
		}

		autoscalerConfig := &AutoscalerConfig{
			Namespace:     config.Namespace,
			RedisAddr:     config.RedisAddr,
			RedisPassword: config.RedisPassword,
			RedisDB:       config.RedisDB,
			CheckInterval: checkInterval,
			MetricsPort:   config.MetricsPort,
			LogLevel:      config.LogLevel,
			STTWorkers:    config.STTWorkers,
			OCRWorkers:    config.OCRWorkers,
		}

		// Set log level
		level, err := logrus.ParseLevel(autoscalerConfig.LogLevel)
		if err != nil {
			level = logrus.InfoLevel
		}
		logrus.SetLevel(level)

		// Create controller
		controller, err := NewAutoscalerController(autoscalerConfig)
		if err != nil {
			logrus.Fatalf("Failed to create controller: %v", err)
		}

		// Start metrics server
		go func() {
			http.Handle("/metrics", promhttp.Handler())
			http.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) {
				w.WriteHeader(http.StatusOK)
				w.Write([]byte("OK"))
			})
			
			logrus.Infof("Starting metrics server on port %d", autoscalerConfig.MetricsPort)
			if err := http.ListenAndServe(fmt.Sprintf(":%d", autoscalerConfig.MetricsPort), nil); err != nil {
				logrus.WithError(err).Error("Metrics server failed")
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

		// Start controller
		logrus.Info("Starting autoscaler controller")
		if err := controller.Run(ctx); err != nil && err != context.Canceled {
			logrus.Fatalf("Controller failed: %v", err)
		}

		logrus.Info("Autoscaler controller stopped")
	},
}

func init() {
	// Set default configuration
	viper.SetDefault("namespace", "streamprocess")
	viper.SetDefault("redis_addr", "localhost:6379")
	viper.SetDefault("redis_password", "")
	viper.SetDefault("redis_db", 0)
	viper.SetDefault("check_interval", "30s")
	viper.SetDefault("metrics_port", 9093)
	viper.SetDefault("log_level", "info")

	// STT worker defaults
	viper.SetDefault("stt_workers.deployment_name", "stt-worker")
	viper.SetDefault("stt_workers.min_replicas", 1)
	viper.SetDefault("stt_workers.max_replicas", 10)
	viper.SetDefault("stt_workers.target_queue_depth", 10)
	viper.SetDefault("stt_workers.target_cpu", 0.7)
	viper.SetDefault("stt_workers.target_memory", 0.8)
	viper.SetDefault("stt_workers.kp", 0.5)
	viper.SetDefault("stt_workers.ki", 0.1)
	viper.SetDefault("stt_workers.kd", 0.05)

	// OCR worker defaults
	viper.SetDefault("ocr_workers.deployment_name", "ocr-worker")
	viper.SetDefault("ocr_workers.min_replicas", 1)
	viper.SetDefault("ocr_workers.max_replicas", 10)
	viper.SetDefault("ocr_workers.target_queue_depth", 5)
	viper.SetDefault("ocr_workers.target_cpu", 0.7)
	viper.SetDefault("ocr_workers.target_memory", 0.8)
	viper.SetDefault("ocr_workers.kp", 0.6)
	viper.SetDefault("ocr_workers.ki", 0.15)
	viper.SetDefault("ocr_workers.kd", 0.1)

	// Bind environment variables
	viper.AutomaticEnv()
	viper.SetEnvPrefix("AUTOSCALER")

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