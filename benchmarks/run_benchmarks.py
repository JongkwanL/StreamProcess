"""Comprehensive benchmarking suite for StreamProcess."""

import asyncio
import time
import statistics
import json
import argparse
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import grpc
import numpy as np
from loguru import logger
import matplotlib.pyplot as plt
import seaborn as sns
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
import websockets

from src.config import settings
from src.generated import stream_process_pb2, stream_process_pb2_grpc
from .load_generator import LoadGenerator
from .metrics_collector import MetricsCollector


@dataclass
class BenchmarkResult:
    """Benchmark result data structure."""
    test_name: str
    total_requests: int
    successful_requests: int
    failed_requests: int
    avg_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    throughput_rps: float
    error_rate: float
    duration_seconds: float
    cpu_usage_avg: float
    memory_usage_avg: float
    queue_depth_avg: float
    timestamp: float


class StreamProcessBenchmark:
    """Main benchmarking class for StreamProcess."""
    
    def __init__(self, grpc_endpoint: str = "localhost:50051", rest_endpoint: str = "http://localhost:8000"):
        self.grpc_endpoint = grpc_endpoint
        self.rest_endpoint = rest_endpoint
        self.load_generator = LoadGenerator()
        self.metrics_collector = MetricsCollector()
        self.results: List[BenchmarkResult] = []
    
    async def run_all_benchmarks(self) -> List[BenchmarkResult]:
        """Run all benchmark suites."""
        logger.info("Starting comprehensive benchmark suite...")
        
        benchmarks = [
            ("STT Latency Test", self.benchmark_stt_latency),
            ("STT Throughput Test", self.benchmark_stt_throughput),
            ("STT Streaming Test", self.benchmark_stt_streaming),
            ("OCR Latency Test", self.benchmark_ocr_latency),
            ("OCR Throughput Test", self.benchmark_ocr_throughput),
            ("Queue Performance Test", self.benchmark_queue_performance),
            ("Concurrent Users Test", self.benchmark_concurrent_users),
            ("Load Test", self.benchmark_load_test),
            ("Stress Test", self.benchmark_stress_test),
        ]
        
        for test_name, benchmark_func in benchmarks:
            try:
                logger.info(f"Running {test_name}...")
                result = await benchmark_func()
                self.results.append(result)
                logger.info(f"Completed {test_name}: {result.throughput_rps:.2f} RPS, {result.avg_latency_ms:.2f}ms avg latency")
            except Exception as e:
                logger.error(f"Failed to run {test_name}: {e}")
                
        return self.results
    
    async def benchmark_stt_latency(self) -> BenchmarkResult:
        """Benchmark STT latency with single requests."""
        test_name = "STT Latency"
        num_requests = 100
        latencies = []
        successful = 0
        failed = 0
        
        # Generate test audio
        audio_data = self.load_generator.generate_audio(duration_seconds=1.0)
        
        start_time = time.time()
        
        async with grpc.aio.insecure_channel(self.grpc_endpoint) as channel:
            stub = stream_process_pb2_grpc.STTServiceStub(channel)
            
            for i in range(num_requests):
                request_start = time.time()
                
                try:
                    request = stream_process_pb2.RecognizeRequest(
                        audio_config=stream_process_pb2.AudioConfig(
                            encoding=stream_process_pb2.AudioConfig.LINEAR16,
                            sample_rate_hertz=16000,
                            language_code="en"
                        ),
                        recognition_config=stream_process_pb2.RecognitionConfig(
                            model="base",
                            beam_size=1
                        ),
                        audio_content=audio_data
                    )
                    
                    response = await stub.Recognize(request)
                    
                    if response.status == stream_process_pb2.STATUS_COMPLETED:
                        successful += 1
                    else:
                        failed += 1
                    
                except Exception as e:
                    logger.debug(f"Request {i} failed: {e}")
                    failed += 1
                
                latency = (time.time() - request_start) * 1000
                latencies.append(latency)
        
        duration = time.time() - start_time
        
        return self._calculate_result(
            test_name, num_requests, successful, failed, latencies, duration
        )
    
    async def benchmark_stt_throughput(self) -> BenchmarkResult:
        """Benchmark STT throughput with concurrent requests."""
        test_name = "STT Throughput"
        num_requests = 1000
        concurrency = 50
        latencies = []
        successful = 0
        failed = 0
        
        # Generate test audio
        audio_data = self.load_generator.generate_audio(duration_seconds=1.0)
        
        start_time = time.time()
        
        async def make_request(session_id: int):
            try:
                async with grpc.aio.insecure_channel(self.grpc_endpoint) as channel:
                    stub = stream_process_pb2_grpc.STTServiceStub(channel)
                    
                    request_start = time.time()
                    
                    request = stream_process_pb2.RecognizeRequest(
                        audio_config=stream_process_pb2.AudioConfig(
                            encoding=stream_process_pb2.AudioConfig.LINEAR16,
                            sample_rate_hertz=16000,
                            language_code="en"
                        ),
                        recognition_config=stream_process_pb2.RecognitionConfig(
                            model="base",
                            beam_size=1
                        ),
                        audio_content=audio_data
                    )
                    
                    response = await stub.Recognize(request)
                    latency = (time.time() - request_start) * 1000
                    
                    return response.status == stream_process_pb2.STATUS_COMPLETED, latency
                    
            except Exception as e:
                latency = (time.time() - request_start) * 1000
                return False, latency
        
        # Run concurrent requests
        tasks = [make_request(i) for i in range(num_requests)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for result in results:
            if isinstance(result, Exception):
                failed += 1
                latencies.append(5000)  # Timeout latency
            else:
                success, latency = result
                if success:
                    successful += 1
                else:
                    failed += 1
                latencies.append(latency)
        
        duration = time.time() - start_time
        
        return self._calculate_result(
            test_name, num_requests, successful, failed, latencies, duration
        )
    
    async def benchmark_stt_streaming(self) -> BenchmarkResult:
        """Benchmark STT streaming performance."""
        test_name = "STT Streaming"
        num_streams = 20
        stream_duration = 10  # seconds
        latencies = []
        successful = 0
        failed = 0
        
        start_time = time.time()
        
        async def streaming_session(session_id: int):
            try:
                async with grpc.aio.insecure_channel(self.grpc_endpoint) as channel:
                    stub = stream_process_pb2_grpc.STTServiceStub(channel)
                    
                    async def request_generator():
                        # Send config
                        config = stream_process_pb2.StreamingRecognitionConfig(
                            audio_config=stream_process_pb2.AudioConfig(
                                encoding=stream_process_pb2.AudioConfig.LINEAR16,
                                sample_rate_hertz=16000,
                                language_code="en"
                            ),
                            recognition_config=stream_process_pb2.RecognitionConfig(
                                model="base",
                                beam_size=1
                            ),
                            enable_partial_results=True,
                            partial_results_interval_ms=150
                        )
                        
                        yield stream_process_pb2.StreamingRecognizeRequest(config=config)
                        
                        # Stream audio chunks
                        for i in range(stream_duration * 10):  # 100ms chunks
                            audio_chunk = self.load_generator.generate_audio_chunk(duration_ms=100)
                            
                            yield stream_process_pb2.StreamingRecognizeRequest(
                                audio_chunk=stream_process_pb2.AudioChunk(
                                    content=audio_chunk,
                                    offset_ms=i * 100,
                                    duration_ms=100
                                )
                            )
                            
                            await asyncio.sleep(0.1)  # Real-time simulation
                    
                    session_start = time.time()
                    response_count = 0
                    
                    async for response in stub.StreamingRecognize(request_generator()):
                        response_latency = (time.time() - session_start) * 1000
                        latencies.append(response_latency)
                        response_count += 1
                    
                    return True, response_count
                    
            except Exception as e:
                logger.debug(f"Streaming session {session_id} failed: {e}")
                return False, 0
        
        # Run concurrent streaming sessions
        tasks = [streaming_session(i) for i in range(num_streams)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        total_responses = 0
        for result in results:
            if isinstance(result, Exception):
                failed += 1
            else:
                success, response_count = result
                if success:
                    successful += 1
                    total_responses += response_count
                else:
                    failed += 1
        
        duration = time.time() - start_time
        
        return self._calculate_result(
            test_name, total_responses, successful, failed, latencies, duration
        )
    
    async def benchmark_ocr_latency(self) -> BenchmarkResult:
        """Benchmark OCR latency."""
        test_name = "OCR Latency"
        num_requests = 100
        latencies = []
        successful = 0
        failed = 0
        
        # Generate test image
        image_data = self.load_generator.generate_test_image()
        
        start_time = time.time()
        
        async with grpc.aio.insecure_channel(self.grpc_endpoint) as channel:
            stub = stream_process_pb2_grpc.OCRServiceStub(channel)
            
            for i in range(num_requests):
                request_start = time.time()
                
                try:
                    request = stream_process_pb2.DocumentRequest(
                        config=stream_process_pb2.DocumentConfig(
                            languages=["en"],
                            detect_layout=True
                        ),
                        image_content=image_data
                    )
                    
                    response = await stub.ProcessDocument(request)
                    
                    if response.status == stream_process_pb2.STATUS_COMPLETED:
                        successful += 1
                    else:
                        failed += 1
                    
                except Exception as e:
                    logger.debug(f"OCR request {i} failed: {e}")
                    failed += 1
                
                latency = (time.time() - request_start) * 1000
                latencies.append(latency)
        
        duration = time.time() - start_time
        
        return self._calculate_result(
            test_name, num_requests, successful, failed, latencies, duration
        )
    
    async def benchmark_ocr_throughput(self) -> BenchmarkResult:
        """Benchmark OCR throughput with concurrent requests."""
        test_name = "OCR Throughput"
        num_requests = 500
        concurrency = 25
        latencies = []
        successful = 0
        failed = 0
        
        # Generate test image
        image_data = self.load_generator.generate_test_image()
        
        start_time = time.time()
        
        async def make_ocr_request():
            try:
                async with grpc.aio.insecure_channel(self.grpc_endpoint) as channel:
                    stub = stream_process_pb2_grpc.OCRServiceStub(channel)
                    
                    request_start = time.time()
                    
                    request = stream_process_pb2.DocumentRequest(
                        config=stream_process_pb2.DocumentConfig(
                            languages=["en"],
                            detect_layout=True
                        ),
                        image_content=image_data
                    )
                    
                    response = await stub.ProcessDocument(request)
                    latency = (time.time() - request_start) * 1000
                    
                    return response.status == stream_process_pb2.STATUS_COMPLETED, latency
                    
            except Exception as e:
                latency = (time.time() - request_start) * 1000
                return False, latency
        
        # Create semaphore for concurrency control
        semaphore = asyncio.Semaphore(concurrency)
        
        async def limited_request():
            async with semaphore:
                return await make_ocr_request()
        
        # Run concurrent requests
        tasks = [limited_request() for _ in range(num_requests)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for result in results:
            if isinstance(result, Exception):
                failed += 1
                latencies.append(10000)  # Timeout latency
            else:
                success, latency = result
                if success:
                    successful += 1
                else:
                    failed += 1
                latencies.append(latency)
        
        duration = time.time() - start_time
        
        return self._calculate_result(
            test_name, num_requests, successful, failed, latencies, duration
        )
    
    async def benchmark_queue_performance(self) -> BenchmarkResult:
        """Benchmark queue performance."""
        test_name = "Queue Performance"
        
        # This would require access to the queue directly
        # For now, simulate queue performance through API
        return await self.benchmark_stt_throughput()  # Placeholder
    
    async def benchmark_concurrent_users(self) -> BenchmarkResult:
        """Benchmark with increasing concurrent users."""
        test_name = "Concurrent Users"
        max_users = 100
        latencies = []
        successful = 0
        failed = 0
        total_requests = 0
        
        start_time = time.time()
        
        # Gradually increase load
        for users in range(10, max_users + 1, 10):
            logger.info(f"Testing with {users} concurrent users...")
            
            # Generate requests for this user level
            requests_per_user = 10
            user_requests = users * requests_per_user
            total_requests += user_requests
            
            # Run concurrent requests
            async def make_request():
                try:
                    response = requests.post(
                        f"{self.rest_endpoint}/stt/upload",
                        files={"file": ("test.wav", self.load_generator.generate_audio(1.0), "audio/wav")},
                        timeout=30
                    )
                    return response.status_code == 200, response.elapsed.total_seconds() * 1000
                except Exception as e:
                    return False, 5000
            
            tasks = [make_request() for _ in range(user_requests)]
            results = await asyncio.gather(*[asyncio.create_task(asyncio.to_thread(task)) for task in tasks])
            
            for success, latency in results:
                if success:
                    successful += 1
                else:
                    failed += 1
                latencies.append(latency)
            
            # Brief pause between user level tests
            await asyncio.sleep(2)
        
        duration = time.time() - start_time
        
        return self._calculate_result(
            test_name, total_requests, successful, failed, latencies, duration
        )
    
    async def benchmark_load_test(self) -> BenchmarkResult:
        """Sustained load test."""
        test_name = "Load Test"
        duration_minutes = 5
        requests_per_second = 50
        
        latencies = []
        successful = 0
        failed = 0
        total_requests = 0
        
        start_time = time.time()
        end_time = start_time + (duration_minutes * 60)
        
        async def request_worker():
            nonlocal successful, failed, total_requests
            
            while time.time() < end_time:
                try:
                    request_start = time.time()
                    
                    # Alternate between STT and OCR requests
                    if total_requests % 2 == 0:
                        response = requests.post(
                            f"{self.rest_endpoint}/stt/upload",
                            files={"file": ("test.wav", self.load_generator.generate_audio(1.0), "audio/wav")},
                            timeout=10
                        )
                    else:
                        response = requests.post(
                            f"{self.rest_endpoint}/ocr/process",
                            files={"file": ("test.png", self.load_generator.generate_test_image(), "image/png")},
                            timeout=10
                        )
                    
                    latency = (time.time() - request_start) * 1000
                    latencies.append(latency)
                    total_requests += 1
                    
                    if response.status_code == 200:
                        successful += 1
                    else:
                        failed += 1
                    
                    # Rate limiting
                    await asyncio.sleep(1.0 / requests_per_second)
                    
                except Exception as e:
                    failed += 1
                    total_requests += 1
                    latencies.append(10000)
        
        # Run multiple workers
        workers = min(requests_per_second, 50)  # Cap workers
        tasks = [request_worker() for _ in range(workers)]
        await asyncio.gather(*tasks)
        
        actual_duration = time.time() - start_time
        
        return self._calculate_result(
            test_name, total_requests, successful, failed, latencies, actual_duration
        )
    
    async def benchmark_stress_test(self) -> BenchmarkResult:
        """Stress test to find breaking point."""
        test_name = "Stress Test"
        
        latencies = []
        successful = 0
        failed = 0
        total_requests = 0
        
        start_time = time.time()
        
        # Gradually increase load until failure
        for rps in range(10, 1000, 20):
            logger.info(f"Stress testing at {rps} RPS...")
            
            # Test for 30 seconds at this rate
            test_duration = 30
            batch_requests = rps * test_duration
            
            async def stress_request():
                try:
                    request_start = time.time()
                    response = requests.post(
                        f"{self.rest_endpoint}/stt/upload",
                        files={"file": ("test.wav", self.load_generator.generate_audio(0.5), "audio/wav")},
                        timeout=5
                    )
                    latency = (time.time() - request_start) * 1000
                    return response.status_code == 200, latency
                except Exception:
                    return False, 5000
            
            # Run batch
            tasks = [stress_request() for _ in range(batch_requests)]
            results = await asyncio.gather(*[asyncio.to_thread(task) for task in tasks])
            
            batch_successful = sum(1 for success, _ in results if success)
            batch_failed = len(results) - batch_successful
            
            successful += batch_successful
            failed += batch_failed
            total_requests += len(results)
            
            for success, latency in results:
                latencies.append(latency)
            
            # Check if system is failing
            error_rate = batch_failed / len(results)
            avg_latency = statistics.mean([lat for _, lat in results])
            
            if error_rate > 0.1 or avg_latency > 10000:  # 10% error rate or 10s latency
                logger.info(f"Breaking point reached at {rps} RPS")
                break
        
        duration = time.time() - start_time
        
        return self._calculate_result(
            test_name, total_requests, successful, failed, latencies, duration
        )
    
    def _calculate_result(
        self,
        test_name: str,
        total_requests: int,
        successful: int,
        failed: int,
        latencies: List[float],
        duration: float
    ) -> BenchmarkResult:
        """Calculate benchmark result from raw data."""
        if not latencies:
            latencies = [0]
        
        return BenchmarkResult(
            test_name=test_name,
            total_requests=total_requests,
            successful_requests=successful,
            failed_requests=failed,
            avg_latency_ms=statistics.mean(latencies),
            p50_latency_ms=statistics.median(latencies),
            p95_latency_ms=np.percentile(latencies, 95),
            p99_latency_ms=np.percentile(latencies, 99),
            throughput_rps=successful / duration if duration > 0 else 0,
            error_rate=failed / total_requests if total_requests > 0 else 0,
            duration_seconds=duration,
            cpu_usage_avg=0.0,  # Would need system monitoring
            memory_usage_avg=0.0,  # Would need system monitoring
            queue_depth_avg=0.0,  # Would need queue monitoring
            timestamp=time.time()
        )
    
    def generate_report(self, output_dir: str = "benchmark_results"):
        """Generate comprehensive benchmark report."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Save raw results
        results_file = output_path / "results.json"
        with open(results_file, 'w') as f:
            json.dump([asdict(result) for result in self.results], f, indent=2)
        
        # Generate plots
        self._generate_plots(output_path)
        
        # Generate HTML report
        self._generate_html_report(output_path)
        
        logger.info(f"Benchmark report generated in {output_path}")
    
    def _generate_plots(self, output_path: Path):
        """Generate visualization plots."""
        if not self.results:
            return
        
        plt.style.use('seaborn-v0_8')
        
        # Latency comparison
        plt.figure(figsize=(12, 6))
        test_names = [r.test_name for r in self.results]
        avg_latencies = [r.avg_latency_ms for r in self.results]
        p99_latencies = [r.p99_latency_ms for r in self.results]
        
        x = np.arange(len(test_names))
        width = 0.35
        
        plt.bar(x - width/2, avg_latencies, width, label='Average Latency', alpha=0.8)
        plt.bar(x + width/2, p99_latencies, width, label='P99 Latency', alpha=0.8)
        
        plt.xlabel('Test Name')
        plt.ylabel('Latency (ms)')
        plt.title('Latency Comparison')
        plt.xticks(x, test_names, rotation=45)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_path / 'latency_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Throughput comparison
        plt.figure(figsize=(10, 6))
        throughputs = [r.throughput_rps for r in self.results]
        
        plt.bar(test_names, throughputs, alpha=0.8, color='skyblue')
        plt.xlabel('Test Name')
        plt.ylabel('Throughput (RPS)')
        plt.title('Throughput Comparison')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(output_path / 'throughput_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Error rate comparison
        plt.figure(figsize=(10, 6))
        error_rates = [r.error_rate * 100 for r in self.results]
        
        plt.bar(test_names, error_rates, alpha=0.8, color='salmon')
        plt.xlabel('Test Name')
        plt.ylabel('Error Rate (%)')
        plt.title('Error Rate Comparison')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(output_path / 'error_rate_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _generate_html_report(self, output_path: Path):
        """Generate HTML report."""
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>StreamProcess Benchmark Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .metric {{ font-weight: bold; color: #2c3e50; }}
                .plot {{ margin: 20px 0; text-align: center; }}
                .summary {{ background-color: #f8f9fa; padding: 20px; border-radius: 5px; margin: 20px 0; }}
            </style>
        </head>
        <body>
            <h1>StreamProcess Benchmark Report</h1>
            <p>Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}</p>
            
            <div class="summary">
                <h2>Summary</h2>
                <p>Total tests: {len(self.results)}</p>
                <p>Best throughput: {max(r.throughput_rps for r in self.results):.2f} RPS</p>
                <p>Best latency: {min(r.avg_latency_ms for r in self.results):.2f} ms</p>
                <p>Overall error rate: {statistics.mean([r.error_rate for r in self.results]) * 100:.2f}%</p>
            </div>
            
            <h2>Detailed Results</h2>
            <table>
                <tr>
                    <th>Test Name</th>
                    <th>Throughput (RPS)</th>
                    <th>Avg Latency (ms)</th>
                    <th>P99 Latency (ms)</th>
                    <th>Error Rate (%)</th>
                    <th>Success Rate (%)</th>
                </tr>
        """
        
        for result in self.results:
            success_rate = (result.successful_requests / result.total_requests * 100) if result.total_requests > 0 else 0
            html_content += f"""
                <tr>
                    <td>{result.test_name}</td>
                    <td class="metric">{result.throughput_rps:.2f}</td>
                    <td class="metric">{result.avg_latency_ms:.2f}</td>
                    <td class="metric">{result.p99_latency_ms:.2f}</td>
                    <td class="metric">{result.error_rate * 100:.2f}</td>
                    <td class="metric">{success_rate:.2f}</td>
                </tr>
            """
        
        html_content += """
            </table>
            
            <div class="plot">
                <h3>Latency Comparison</h3>
                <img src="latency_comparison.png" alt="Latency Comparison">
            </div>
            
            <div class="plot">
                <h3>Throughput Comparison</h3>
                <img src="throughput_comparison.png" alt="Throughput Comparison">
            </div>
            
            <div class="plot">
                <h3>Error Rate Comparison</h3>
                <img src="error_rate_comparison.png" alt="Error Rate Comparison">
            </div>
            
        </body>
        </html>
        """
        
        with open(output_path / 'report.html', 'w') as f:
            f.write(html_content)


async def main():
    """Main entry point for benchmarks."""
    parser = argparse.ArgumentParser(description='StreamProcess Benchmark Suite')
    parser.add_argument('--grpc-endpoint', default='localhost:50051', help='gRPC endpoint')
    parser.add_argument('--rest-endpoint', default='http://localhost:8000', help='REST endpoint')
    parser.add_argument('--output-dir', default='benchmark_results', help='Output directory')
    parser.add_argument('--tests', nargs='+', help='Specific tests to run')
    
    args = parser.parse_args()
    
    benchmark = StreamProcessBenchmark(args.grpc_endpoint, args.rest_endpoint)
    
    try:
        results = await benchmark.run_all_benchmarks()
        benchmark.generate_report(args.output_dir)
        
        # Print summary
        print("\n" + "="*60)
        print("BENCHMARK SUMMARY")
        print("="*60)
        for result in results:
            print(f"{result.test_name:30} | {result.throughput_rps:8.2f} RPS | {result.avg_latency_ms:8.2f}ms")
        print("="*60)
        
    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(asyncio.run(main()))