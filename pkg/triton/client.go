package triton

import (
	"context"
	"fmt"
	"time"

	"github.com/sirupsen/logrus"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
	
	tritongrpc "github.com/triton-inference-server/client/src/grpc_generated/go/grpc-client"
)

// TritonClient manages connections to Triton Inference Server
type TritonClient struct {
	client tritongrpc.GRPCInferenceServiceClient
	conn   *grpc.ClientConn
	logger *logrus.Logger
}

// InferenceRequest represents a Triton inference request
type InferenceRequest struct {
	ModelName    string
	ModelVersion string
	Inputs       []*tritongrpc.ModelInferRequest_InferInputTensor
	Outputs      []*tritongrpc.ModelInferRequest_InferRequestedOutputTensor
	RawData      [][]byte
}

// InferenceResult represents a Triton inference result
type InferenceResult struct {
	Outputs []*tritongrpc.ModelInferResponse_InferOutputTensor
	RawData [][]byte
}

// NewTritonClient creates a new Triton client
func NewTritonClient(serverURL string) (*TritonClient, error) {
	conn, err := grpc.Dial(serverURL, grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		return nil, fmt.Errorf("failed to connect to Triton server: %w", err)
	}

	client := tritongrpc.NewGRPCInferenceServiceClient(conn)
	
	return &TritonClient{
		client: client,
		conn:   conn,
		logger: logrus.New(),
	}, nil
}

// Close closes the connection to Triton server
func (tc *TritonClient) Close() error {
	return tc.conn.Close()
}

// ServerLive checks if Triton server is live
func (tc *TritonClient) ServerLive(ctx context.Context) (bool, error) {
	req := &tritongrpc.ServerLiveRequest{}
	resp, err := tc.client.ServerLive(ctx, req)
	if err != nil {
		return false, err
	}
	return resp.Live, nil
}

// ServerReady checks if Triton server is ready
func (tc *TritonClient) ServerReady(ctx context.Context) (bool, error) {
	req := &tritongrpc.ServerReadyRequest{}
	resp, err := tc.client.ServerReady(ctx, req)
	if err != nil {
		return false, err
	}
	return resp.Ready, nil
}

// ModelReady checks if a specific model is ready
func (tc *TritonClient) ModelReady(ctx context.Context, modelName, modelVersion string) (bool, error) {
	req := &tritongrpc.ModelReadyRequest{
		Name:    modelName,
		Version: modelVersion,
	}
	resp, err := tc.client.ModelReady(ctx, req)
	if err != nil {
		return false, err
	}
	return resp.Ready, nil
}

// GetModelMetadata gets metadata for a model
func (tc *TritonClient) GetModelMetadata(ctx context.Context, modelName, modelVersion string) (*tritongrpc.ModelMetadataResponse, error) {
	req := &tritongrpc.ModelMetadataRequest{
		Name:    modelName,
		Version: modelVersion,
	}
	return tc.client.ModelMetadata(ctx, req)
}

// GetModelConfig gets configuration for a model
func (tc *TritonClient) GetModelConfig(ctx context.Context, modelName, modelVersion string) (*tritongrpc.ModelConfigResponse, error) {
	req := &tritongrpc.ModelConfigRequest{
		Name:    modelName,
		Version: modelVersion,
	}
	return tc.client.ModelConfig(ctx, req)
}

// Infer performs inference on Triton server
func (tc *TritonClient) Infer(ctx context.Context, req *InferenceRequest) (*InferenceResult, error) {
	start := time.Now()

	// Create Triton inference request
	tritonReq := &tritongrpc.ModelInferRequest{
		ModelName:    req.ModelName,
		ModelVersion: req.ModelVersion,
		Inputs:       req.Inputs,
		Outputs:      req.Outputs,
		RawInputs:    req.RawData,
	}

	// Perform inference
	resp, err := tc.client.ModelInfer(ctx, tritonReq)
	if err != nil {
		return nil, fmt.Errorf("inference failed: %w", err)
	}

	tc.logger.WithFields(logrus.Fields{
		"model":    req.ModelName,
		"version":  req.ModelVersion,
		"duration": time.Since(start),
	}).Debug("Inference completed")

	return &InferenceResult{
		Outputs: resp.Outputs,
		RawData: resp.RawOutputs,
	}, nil
}

// WhisperSTTClient provides high-level STT inference via Triton
type WhisperSTTClient struct {
	triton *TritonClient
	logger *logrus.Logger
}

// NewWhisperSTTClient creates a new Whisper STT client
func NewWhisperSTTClient(tritonURL string) (*WhisperSTTClient, error) {
	triton, err := NewTritonClient(tritonURL)
	if err != nil {
		return nil, err
	}

	return &WhisperSTTClient{
		triton: triton,
		logger: logrus.New(),
	}, nil
}

// TranscribeAudio transcribes audio using Whisper ONNX model
func (w *WhisperSTTClient) TranscribeAudio(ctx context.Context, melSpectrogram []float32) ([]int32, error) {
	// Prepare input tensor
	inputTensor := &tritongrpc.ModelInferRequest_InferInputTensor{
		Name:     "mel",
		Datatype: "FP32",
		Shape:    []int64{1, 80, int64(len(melSpectrogram) / 80)}, // Batch=1, Mel=80, Time=dynamic
	}

	// Prepare output tensor request
	outputTensor := &tritongrpc.ModelInferRequest_InferRequestedOutputTensor{
		Name: "output_ids",
	}

	// Convert mel spectrogram to bytes
	inputData := make([]byte, len(melSpectrogram)*4) // 4 bytes per float32
	for i, val := range melSpectrogram {
		bits := *(*uint32)(unsafe.Pointer(&val))
		inputData[i*4] = byte(bits)
		inputData[i*4+1] = byte(bits >> 8)
		inputData[i*4+2] = byte(bits >> 16)
		inputData[i*4+3] = byte(bits >> 24)
	}

	// Create inference request
	req := &InferenceRequest{
		ModelName:    "whisper_onnx",
		ModelVersion: "",
		Inputs:       []*tritongrpc.ModelInferRequest_InferInputTensor{inputTensor},
		Outputs:      []*tritongrpc.ModelInferRequest_InferRequestedOutputTensor{outputTensor},
		RawData:      [][]byte{inputData},
	}

	// Perform inference
	result, err := w.triton.Infer(ctx, req)
	if err != nil {
		return nil, fmt.Errorf("Whisper inference failed: %w", err)
	}

	// Parse output tokens
	if len(result.RawData) == 0 {
		return nil, fmt.Errorf("no output data received")
	}

	outputData := result.RawData[0]
	tokenCount := len(outputData) / 4 // 4 bytes per int32
	tokens := make([]int32, tokenCount)

	for i := 0; i < tokenCount; i++ {
		bytes := outputData[i*4 : (i+1)*4]
		tokens[i] = int32(bytes[0]) | int32(bytes[1])<<8 | int32(bytes[2])<<16 | int32(bytes[3])<<24
	}

	return tokens, nil
}

// Close closes the STT client
func (w *WhisperSTTClient) Close() error {
	return w.triton.Close()
}

// PaddleOCRClient provides high-level OCR inference via Triton
type PaddleOCRClient struct {
	triton *TritonClient
	logger *logrus.Logger
}

// NewPaddleOCRClient creates a new PaddleOCR client
func NewPaddleOCRClient(tritonURL string) (*PaddleOCRClient, error) {
	triton, err := NewTritonClient(tritonURL)
	if err != nil {
		return nil, err
	}

	return &PaddleOCRClient{
		triton: triton,
		logger: logrus.New(),
	}, nil
}

// RecognizeText performs OCR using PaddleOCR ONNX model
func (p *PaddleOCRClient) RecognizeText(ctx context.Context, imageData []float32) ([]float32, error) {
	// Prepare input tensor
	inputTensor := &tritongrpc.ModelInferRequest_InferInputTensor{
		Name:     "x",
		Datatype: "FP32",
		Shape:    []int64{1, 3, 48, 320}, // Batch=1, Channels=3, Height=48, Width=320
	}

	// Prepare output tensor request
	outputTensor := &tritongrpc.ModelInferRequest_InferRequestedOutputTensor{
		Name: "save_infer_model/scale_0.tmp_1",
	}

	// Convert image data to bytes
	inputData := make([]byte, len(imageData)*4) // 4 bytes per float32
	for i, val := range imageData {
		bits := *(*uint32)(unsafe.Pointer(&val))
		inputData[i*4] = byte(bits)
		inputData[i*4+1] = byte(bits >> 8)
		inputData[i*4+2] = byte(bits >> 16)
		inputData[i*4+3] = byte(bits >> 24)
	}

	// Create inference request
	req := &InferenceRequest{
		ModelName:    "paddleocr_onnx",
		ModelVersion: "",
		Inputs:       []*tritongrpc.ModelInferRequest_InferInputTensor{inputTensor},
		Outputs:      []*tritongrpc.ModelInferRequest_InferRequestedOutputTensor{outputTensor},
		RawData:      [][]byte{inputData},
	}

	// Perform inference
	result, err := p.triton.Infer(ctx, req)
	if err != nil {
		return nil, fmt.Errorf("PaddleOCR inference failed: %w", err)
	}

	// Parse output probabilities
	if len(result.RawData) == 0 {
		return nil, fmt.Errorf("no output data received")
	}

	outputData := result.RawData[0]
	probCount := len(outputData) / 4 // 4 bytes per float32
	probs := make([]float32, probCount)

	for i := 0; i < probCount; i++ {
		bytes := outputData[i*4 : (i+1)*4]
		bits := uint32(bytes[0]) | uint32(bytes[1])<<8 | uint32(bytes[2])<<16 | uint32(bytes[3])<<24
		probs[i] = *(*float32)(unsafe.Pointer(&bits))
	}

	return probs, nil
}

// Close closes the OCR client
func (p *PaddleOCRClient) Close() error {
	return p.triton.Close()
}