# Load testing script for AutoGen Code Review Bot

import asyncio
import aiohttp
import time
import json
import statistics
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import List, Dict, Any
import argparse


@dataclass
class LoadTestResult:
    """Results from a load test run."""
    total_requests: int
    successful_requests: int
    failed_requests: int
    total_time: float
    requests_per_second: float
    average_response_time: float
    median_response_time: float
    p95_response_time: float
    p99_response_time: float
    error_rate: float


class LoadTester:
    """Load testing utility for the AutoGen Bot."""
    
    def __init__(self, base_url: str = "http://localhost:8080"):
        self.base_url = base_url
        self.results: List[Dict[str, Any]] = []
    
    async def make_request(
        self, 
        session: aiohttp.ClientSession, 
        endpoint: str,
        method: str = "GET",
        data: Dict = None
    ) -> Dict[str, Any]:
        """Make a single HTTP request and measure performance."""
        start_time = time.time()
        
        try:
            async with session.request(
                method=method,
                url=f"{self.base_url}{endpoint}",
                json=data,
                timeout=aiohttp.ClientTimeout(total=60)
            ) as response:
                response_time = time.time() - start_time
                content = await response.text()
                
                return {
                    "success": response.status == 200,
                    "status_code": response.status,
                    "response_time": response_time,
                    "content_length": len(content),
                    "endpoint": endpoint
                }
        except Exception as e:
            response_time = time.time() - start_time
            return {
                "success": False,
                "status_code": 0,
                "response_time": response_time,
                "error": str(e),
                "endpoint": endpoint
            }
    
    async def run_concurrent_requests(
        self,
        endpoint: str,
        concurrent_users: int,
        requests_per_user: int,
        method: str = "GET",
        data: Dict = None
    ) -> LoadTestResult:
        """Run concurrent requests against an endpoint."""
        
        async def user_session():
            """Simulate a single user making multiple requests."""
            async with aiohttp.ClientSession() as session:
                tasks = []
                for _ in range(requests_per_user):
                    task = self.make_request(session, endpoint, method, data)
                    tasks.append(task)
                
                return await asyncio.gather(*tasks)
        
        # Start load test
        start_time = time.time()
        print(f"Starting load test: {concurrent_users} users, {requests_per_user} requests each")
        
        # Create tasks for concurrent users
        user_tasks = [user_session() for _ in range(concurrent_users)]
        user_results = await asyncio.gather(*user_tasks)
        
        # Flatten results
        all_results = []
        for user_result in user_results:
            all_results.extend(user_result)
        
        total_time = time.time() - start_time
        
        # Calculate statistics
        successful_requests = sum(1 for r in all_results if r["success"])
        failed_requests = len(all_results) - successful_requests
        response_times = [r["response_time"] for r in all_results if r["success"]]
        
        if response_times:
            avg_response_time = statistics.mean(response_times)
            median_response_time = statistics.median(response_times)
            p95_response_time = self._percentile(response_times, 95)
            p99_response_time = self._percentile(response_times, 99)
        else:
            avg_response_time = median_response_time = p95_response_time = p99_response_time = 0
        
        return LoadTestResult(
            total_requests=len(all_results),
            successful_requests=successful_requests,
            failed_requests=failed_requests,
            total_time=total_time,
            requests_per_second=len(all_results) / total_time,
            average_response_time=avg_response_time,
            median_response_time=median_response_time,
            p95_response_time=p95_response_time,
            p99_response_time=p99_response_time,
            error_rate=failed_requests / len(all_results) * 100
        )
    
    def _percentile(self, data: List[float], percentile: int) -> float:
        """Calculate percentile value."""
        sorted_data = sorted(data)
        index = (percentile / 100.0) * (len(sorted_data) - 1)
        
        if index.is_integer():
            return sorted_data[int(index)]
        else:
            lower = sorted_data[int(index)]
            upper = sorted_data[int(index) + 1]
            return lower + (upper - lower) * (index - int(index))
    
    async def health_check_load_test(self) -> LoadTestResult:
        """Load test the health check endpoint."""
        return await self.run_concurrent_requests(
            endpoint="/health",
            concurrent_users=10,
            requests_per_user=20
        )
    
    async def webhook_load_test(self) -> LoadTestResult:
        """Load test the webhook endpoint with PR data."""
        pr_webhook_data = {
            "action": "opened",
            "pull_request": {
                "number": 123,
                "title": "Test PR",
                "body": "Load testing PR webhook",
                "base": {"ref": "main"},
                "head": {"ref": "feature-branch", "sha": "abc123"}
            },
            "repository": {
                "full_name": "test/repo",
                "clone_url": "https://github.com/test/repo.git"
            }
        }
        
        return await self.run_concurrent_requests(
            endpoint="/webhook",
            concurrent_users=5,
            requests_per_user=10,
            method="POST",
            data=pr_webhook_data
        )
    
    async def metrics_load_test(self) -> LoadTestResult:
        """Load test the metrics endpoint."""
        return await self.run_concurrent_requests(
            endpoint="/metrics",
            concurrent_users=20,
            requests_per_user=50
        )
    
    def print_results(self, test_name: str, result: LoadTestResult):
        """Print load test results in a readable format."""
        print(f"\n{'='*60}")
        print(f"LOAD TEST RESULTS: {test_name}")
        print(f"{'='*60}")
        print(f"Total Requests:       {result.total_requests}")
        print(f"Successful Requests:  {result.successful_requests}")
        print(f"Failed Requests:      {result.failed_requests}")
        print(f"Error Rate:           {result.error_rate:.2f}%")
        print(f"Total Time:           {result.total_time:.2f}s")
        print(f"Requests/Second:      {result.requests_per_second:.2f}")
        print(f"Average Response:     {result.average_response_time*1000:.2f}ms")
        print(f"Median Response:      {result.median_response_time*1000:.2f}ms")
        print(f"95th Percentile:      {result.p95_response_time*1000:.2f}ms")
        print(f"99th Percentile:      {result.p99_response_time*1000:.2f}ms")
        
        # Performance assertions
        if result.error_rate > 5.0:
            print(f"⚠️  WARNING: High error rate ({result.error_rate:.2f}%)")
        
        if result.average_response_time > 5.0:
            print(f"⚠️  WARNING: High average response time ({result.average_response_time:.2f}s)")
        
        if result.requests_per_second < 10:
            print(f"⚠️  WARNING: Low throughput ({result.requests_per_second:.2f} req/s)")
    
    async def run_full_load_test_suite(self):
        """Run the complete load test suite."""
        print("Starting AutoGen Bot Load Test Suite")
        print(f"Target URL: {self.base_url}")
        
        test_suite = [
            ("Health Check", self.health_check_load_test),
            ("Metrics Endpoint", self.metrics_load_test),
            ("Webhook Processing", self.webhook_load_test),
        ]
        
        results = {}
        
        for test_name, test_func in test_suite:
            try:
                print(f"\nRunning {test_name} load test...")
                result = await test_func()
                results[test_name] = result
                self.print_results(test_name, result)
            except Exception as e:
                print(f"❌ {test_name} failed: {str(e)}")
                results[test_name] = None
        
        # Save results to file
        self.save_results_to_file(results)
        
        return results
    
    def save_results_to_file(self, results: Dict[str, LoadTestResult]):
        """Save load test results to JSON file."""
        serializable_results = {}
        
        for test_name, result in results.items():
            if result:
                serializable_results[test_name] = {
                    "total_requests": result.total_requests,
                    "successful_requests": result.successful_requests,
                    "failed_requests": result.failed_requests,
                    "total_time": result.total_time,
                    "requests_per_second": result.requests_per_second,
                    "average_response_time": result.average_response_time,
                    "median_response_time": result.median_response_time,
                    "p95_response_time": result.p95_response_time,
                    "p99_response_time": result.p99_response_time,
                    "error_rate": result.error_rate,
                    "timestamp": time.time()
                }
            else:
                serializable_results[test_name] = {"error": "Test failed"}
        
        with open("load_test_results.json", "w") as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"\nResults saved to load_test_results.json")


async def main():
    """Main function to run load tests."""
    parser = argparse.ArgumentParser(description="Load test AutoGen Code Review Bot")
    parser.add_argument(
        "--url", 
        default="http://localhost:8080",
        help="Base URL of the bot service"
    )
    parser.add_argument(
        "--test",
        choices=["health", "webhook", "metrics", "all"],
        default="all",
        help="Specific test to run"
    )
    
    args = parser.parse_args()
    
    tester = LoadTester(base_url=args.url)
    
    if args.test == "all":
        await tester.run_full_load_test_suite()
    elif args.test == "health":
        result = await tester.health_check_load_test()
        tester.print_results("Health Check", result)
    elif args.test == "webhook":
        result = await tester.webhook_load_test()
        tester.print_results("Webhook", result)
    elif args.test == "metrics":
        result = await tester.metrics_load_test()
        tester.print_results("Metrics", result)


if __name__ == "__main__":
    asyncio.run(main())