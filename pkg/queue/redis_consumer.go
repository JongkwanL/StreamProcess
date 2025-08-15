package queue

import (
	"context"
	"encoding/json"
	"fmt"
	"strconv"
	"time"

	"github.com/redis/go-redis/v9"
	"github.com/prometheus/client_golang/prometheus"
	"github.com/sirupsen/logrus"
)

// RedisConsumer handles Redis Streams consumption with priority queues
type RedisConsumer struct {
	client         *redis.Client
	consumerGroup  string
	consumerName   string
	priorityQueues map[int]string
	logger         *logrus.Logger
	metrics        *ConsumerMetrics
	ctx            context.Context
	cancel         context.CancelFunc
}

// Job represents a processing job
type Job struct {
	ID          string                 `json:"id"`
	Type        string                 `json:"type"`
	Priority    int                    `json:"priority"`
	Data        map[string]interface{} `json:"data"`
	CreatedAt   time.Time             `json:"created_at"`
	DequeuedAt  *time.Time            `json:"dequeued_at,omitempty"`
	RetryCount  int                   `json:"retry_count"`
	MaxRetries  int                   `json:"max_retries"`
}

// ConsumerMetrics for Prometheus monitoring
type ConsumerMetrics struct {
	JobsProcessed    prometheus.Counter
	JobsSuccessful   prometheus.Counter
	JobsFailed       prometheus.Counter
	ProcessingTime   prometheus.Histogram
	QueueDepth       prometheus.GaugeVec
	ConsumerLag      prometheus.GaugeVec
}

// NewRedisConsumer creates a new Redis consumer
func NewRedisConsumer(redisAddr, password string, db int, consumerGroup, consumerName string) (*RedisConsumer, error) {
	rdb := redis.NewClient(&redis.Options{
		Addr:     redisAddr,
		Password: password,
		DB:       db,
	})

	// Test connection
	ctx := context.Background()
	if err := rdb.Ping(ctx).Err(); err != nil {
		return nil, fmt.Errorf("failed to connect to Redis: %w", err)
	}

	ctx, cancel := context.WithCancel(context.Background())

	// Initialize priority queues
	priorityQueues := map[int]string{
		0: "streamprocess:queue:low",      // LOW
		1: "streamprocess:queue:normal",   // NORMAL
		2: "streamprocess:queue:high",     // HIGH
		3: "streamprocess:queue:realtime", // REALTIME
	}

	// Initialize metrics
	metrics := &ConsumerMetrics{
		JobsProcessed: prometheus.NewCounter(prometheus.CounterOpts{
			Name: "streamprocess_jobs_processed_total",
			Help: "Total number of jobs processed",
		}),
		JobsSuccessful: prometheus.NewCounter(prometheus.CounterOpts{
			Name: "streamprocess_jobs_successful_total",
			Help: "Total number of successful jobs",
		}),
		JobsFailed: prometheus.NewCounter(prometheus.CounterOpts{
			Name: "streamprocess_jobs_failed_total",
			Help: "Total number of failed jobs",
		}),
		ProcessingTime: prometheus.NewHistogram(prometheus.HistogramOpts{
			Name:    "streamprocess_job_processing_duration_seconds",
			Help:    "Job processing duration in seconds",
			Buckets: prometheus.ExponentialBuckets(0.001, 2, 15), // 1ms to ~30s
		}),
		QueueDepth: *prometheus.NewGaugeVec(prometheus.GaugeOpts{
			Name: "streamprocess_queue_depth",
			Help: "Current queue depth by priority",
		}, []string{"priority"}),
		ConsumerLag: *prometheus.NewGaugeVec(prometheus.GaugeOpts{
			Name: "streamprocess_consumer_lag",
			Help: "Consumer lag by stream",
		}, []string{"stream"}),
	}

	// Register metrics
	prometheus.MustRegister(
		metrics.JobsProcessed,
		metrics.JobsSuccessful,
		metrics.JobsFailed,
		metrics.ProcessingTime,
		&metrics.QueueDepth,
		&metrics.ConsumerLag,
	)

	consumer := &RedisConsumer{
		client:         rdb,
		consumerGroup:  consumerGroup,
		consumerName:   consumerName,
		priorityQueues: priorityQueues,
		logger:         logrus.New(),
		metrics:        metrics,
		ctx:            ctx,
		cancel:         cancel,
	}

	// Initialize consumer groups
	if err := consumer.initializeConsumerGroups(); err != nil {
		return nil, fmt.Errorf("failed to initialize consumer groups: %w", err)
	}

	return consumer, nil
}

// initializeConsumerGroups creates consumer groups for all priority queues
func (c *RedisConsumer) initializeConsumerGroups() error {
	for _, streamKey := range c.priorityQueues {
		err := c.client.XGroupCreateMkStream(c.ctx, streamKey, c.consumerGroup, "0").Err()
		if err != nil && err.Error() != "BUSYGROUP Consumer Group name already exists" {
			return fmt.Errorf("failed to create consumer group for stream %s: %w", streamKey, err)
		}
	}
	return nil
}

// ConsumeJobs consumes jobs from priority queues
func (c *RedisConsumer) ConsumeJobs(processor JobProcessor) error {
	c.logger.WithFields(logrus.Fields{
		"consumer_group": c.consumerGroup,
		"consumer_name":  c.consumerName,
	}).Info("Starting job consumption")

	for {
		select {
		case <-c.ctx.Done():
			return c.ctx.Err()
		default:
			if err := c.consumeFromPriorityQueues(processor); err != nil {
				c.logger.WithError(err).Error("Error consuming from priority queues")
				time.Sleep(1 * time.Second)
			}
		}
	}
}

// consumeFromPriorityQueues consumes jobs in priority order
func (c *RedisConsumer) consumeFromPriorityQueues(processor JobProcessor) error {
	// Process queues in priority order (3=REALTIME to 0=LOW)
	for priority := 3; priority >= 0; priority-- {
		streamKey := c.priorityQueues[priority]
		
		// Try to get jobs from this priority queue
		streams, err := c.client.XReadGroup(c.ctx, &redis.XReadGroupArgs{
			Group:    c.consumerGroup,
			Consumer: c.consumerName,
			Streams:  []string{streamKey, ">"},
			Count:    1,
			Block:    100 * time.Millisecond, // Short block to check other priorities
		}).Result()

		if err != nil {
			if err == redis.Nil {
				continue // No messages, try next priority
			}
			return fmt.Errorf("failed to read from stream %s: %w", streamKey, err)
		}

		// Process messages
		for _, stream := range streams {
			for _, message := range stream.Messages {
				if err := c.processMessage(message, streamKey, processor); err != nil {
					c.logger.WithError(err).WithField("message_id", message.ID).Error("Failed to process message")
				}
			}
		}

		// If we found jobs at this priority, don't check lower priorities this round
		if len(streams) > 0 && len(streams[0].Messages) > 0 {
			break
		}
	}

	return nil
}

// processMessage processes a single message
func (c *RedisConsumer) processMessage(msg redis.XMessage, streamKey string, processor JobProcessor) error {
	start := time.Now()
	c.metrics.JobsProcessed.Inc()

	// Parse job data
	jobData, ok := msg.Values["job_data"].(string)
	if !ok {
		return fmt.Errorf("invalid job data format in message %s", msg.ID)
	}

	var job Job
	if err := json.Unmarshal([]byte(jobData), &job); err != nil {
		return fmt.Errorf("failed to unmarshal job data: %w", err)
	}

	job.ID = msg.ID
	now := time.Now()
	job.DequeuedAt = &now

	c.logger.WithFields(logrus.Fields{
		"job_id":   job.ID,
		"job_type": job.Type,
		"priority": job.Priority,
	}).Info("Processing job")

	// Process the job
	result, err := processor.ProcessJob(c.ctx, &job)
	
	// Record processing time
	c.metrics.ProcessingTime.Observe(time.Since(start).Seconds())

	if err != nil {
		c.metrics.JobsFailed.Inc()
		c.logger.WithError(err).WithField("job_id", job.ID).Error("Job processing failed")
		
		// Handle retry logic
		if job.RetryCount < job.MaxRetries {
			return c.requeueWithRetry(streamKey, &job, err)
		} else {
			return c.moveToDeadLetter(streamKey, &job, err)
		}
	}

	c.metrics.JobsSuccessful.Inc()
	
	// Store result
	if err := c.storeResult(job.ID, result); err != nil {
		c.logger.WithError(err).WithField("job_id", job.ID).Warn("Failed to store job result")
	}

	// Acknowledge message
	return c.client.XAck(c.ctx, streamKey, c.consumerGroup, msg.ID).Err()
}

// requeueWithRetry requeues a failed job with incremented retry count
func (c *RedisConsumer) requeueWithRetry(streamKey string, job *Job, processingErr error) error {
	job.RetryCount++
	
	// Add delay based on retry count (exponential backoff)
	delay := time.Duration(job.RetryCount*job.RetryCount) * time.Second
	time.Sleep(delay)

	// Requeue to appropriate priority queue (possibly lower priority)
	newPriority := job.Priority
	if job.RetryCount > 2 {
		newPriority = max(0, job.Priority-1) // Lower priority for repeated failures
	}

	newStreamKey := c.priorityQueues[newPriority]
	jobData, _ := json.Marshal(job)

	return c.client.XAdd(c.ctx, &redis.XAddArgs{
		Stream: newStreamKey,
		Values: map[string]interface{}{
			"job_data": string(jobData),
			"retry":    true,
			"original_error": processingErr.Error(),
		},
	}).Err()
}

// moveToDeadLetter moves failed job to dead letter queue
func (c *RedisConsumer) moveToDeadLetter(streamKey string, job *Job, processingErr error) error {
	deadLetterKey := "streamprocess:dead_letter"
	
	jobData, _ := json.Marshal(job)
	
	return c.client.XAdd(c.ctx, &redis.XAddArgs{
		Stream: deadLetterKey,
		Values: map[string]interface{}{
			"job_data": string(jobData),
			"final_error": processingErr.Error(),
			"failed_at": time.Now().Unix(),
		},
	}).Err()
}

// storeResult stores job processing result
func (c *RedisConsumer) storeResult(jobID string, result interface{}) error {
	resultKey := fmt.Sprintf("streamprocess:result:%s", jobID)
	
	resultData, err := json.Marshal(result)
	if err != nil {
		return err
	}

	return c.client.Set(c.ctx, resultKey, resultData, 1*time.Hour).Err()
}

// UpdateMetrics updates queue depth and consumer lag metrics
func (c *RedisConsumer) UpdateMetrics() {
	for priority, streamKey := range c.priorityQueues {
		// Update queue depth
		length := c.client.XLen(c.ctx, streamKey).Val()
		c.metrics.QueueDepth.WithLabelValues(strconv.Itoa(priority)).Set(float64(length))

		// Update consumer lag
		groupInfo := c.client.XInfoGroups(c.ctx, streamKey).Val()
		for _, group := range groupInfo {
			if group.Name == c.consumerGroup {
				c.metrics.ConsumerLag.WithLabelValues(streamKey).Set(float64(group.Lag))
				break
			}
		}
	}
}

// Close gracefully shuts down the consumer
func (c *RedisConsumer) Close() error {
	c.cancel()
	return c.client.Close()
}

// JobProcessor interface for processing jobs
type JobProcessor interface {
	ProcessJob(ctx context.Context, job *Job) (interface{}, error)
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}