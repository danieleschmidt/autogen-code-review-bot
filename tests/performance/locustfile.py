"""
Locust performance testing configuration for AutoGen Code Review Bot.

This file defines load testing scenarios using Locust to test the system
under various load conditions and identify performance bottlenecks.
"""

import json
import random
import time
from typing import Dict, Any

from locust import HttpUser, task, between, events


class WebhookUser(HttpUser):
    """Simulates GitHub webhook requests to the bot."""
    
    wait_time = between(1, 5)  # Wait 1-5 seconds between requests
    
    def on_start(self):
        """Initialize user session."""
        self.pr_numbers = list(range(1, 1000))
        self.repo_names = [
            "test-org/repo-1",
            "test-org/repo-2", 
            "example-org/project-a",
            "example-org/project-b"
        ]
    
    @task(3)
    def webhook_pr_opened(self):
        """Simulate PR opened webhook."""
        payload = self._create_pr_webhook_payload("opened")
        
        with self.client.post(
            "/webhook",
            json=payload,
            headers=self._get_webhook_headers(),
            catch_response=True
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Unexpected status code: {response.status_code}")
    
    @task(2)
    def webhook_pr_synchronize(self):
        """Simulate PR synchronize webhook (new commits)."""
        payload = self._create_pr_webhook_payload("synchronize")
        
        with self.client.post(
            "/webhook",
            json=payload,
            headers=self._get_webhook_headers(),
            catch_response=True
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Unexpected status code: {response.status_code}")
    
    @task(1)
    def webhook_pr_closed(self):
        """Simulate PR closed webhook."""
        payload = self._create_pr_webhook_payload("closed")
        
        with self.client.post(
            "/webhook",
            json=payload,
            headers=self._get_webhook_headers(),
            catch_response=True
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Unexpected status code: {response.status_code}")
    
    @task(1)
    def health_check(self):
        """Check health endpoint."""
        with self.client.get("/health", catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Health check failed: {response.status_code}")
    
    @task(1)
    def metrics_endpoint(self):
        """Check metrics endpoint."""
        with self.client.get("/metrics", catch_response=True) as response:
            if response.status_code in [200, 404]:  # 404 if not implemented
                response.success()
            else:
                response.failure(f"Metrics endpoint failed: {response.status_code}")
    
    def _create_pr_webhook_payload(self, action: str) -> Dict[str, Any]:
        """Create a realistic GitHub webhook payload."""
        pr_number = random.choice(self.pr_numbers)
        repo_name = random.choice(self.repo_names)
        owner, repo = repo_name.split("/")
        
        return {
            "action": action,
            "number": pr_number,
            "pull_request": {
                "id": pr_number * 1000 + random.randint(1, 999),
                "number": pr_number,
                "title": f"Test PR #{pr_number}",
                "body": "This is a test pull request for load testing.",
                "state": "open" if action != "closed" else "closed",
                "head": {
                    "sha": self._generate_sha(),
                    "ref": f"feature/test-{pr_number}",
                    "repo": {
                        "name": repo,
                        "full_name": repo_name
                    }
                },
                "base": {
                    "sha": self._generate_sha(),
                    "ref": "main",
                    "repo": {
                        "name": repo,
                        "full_name": repo_name
                    }
                },
                "user": {
                    "login": f"testuser{random.randint(1, 100)}",
                    "id": random.randint(1000, 9999)
                },
                "created_at": "2025-07-27T12:00:00Z",
                "updated_at": "2025-07-27T12:05:00Z"
            },
            "repository": {
                "id": random.randint(10000, 99999),
                "name": repo,
                "full_name": repo_name,
                "owner": {
                    "login": owner,
                    "id": random.randint(1000, 9999)
                },
                "private": False,
                "default_branch": "main"
            },
            "sender": {
                "login": f"testuser{random.randint(1, 100)}",
                "id": random.randint(1000, 9999)
            }
        }
    
    def _get_webhook_headers(self) -> Dict[str, str]:
        """Generate appropriate webhook headers."""
        return {
            "Content-Type": "application/json",
            "X-GitHub-Event": "pull_request",
            "X-GitHub-Delivery": self._generate_sha(),
            "User-Agent": "GitHub-Hookshot/locust-test"
        }
    
    def _generate_sha(self) -> str:
        """Generate a random SHA-like string."""
        import hashlib
        return hashlib.sha1(str(random.random()).encode()).hexdigest()


class APIUser(HttpUser):
    """Simulates API requests to the bot."""
    
    wait_time = between(2, 8)  # Wait 2-8 seconds between requests
    
    def on_start(self):
        """Initialize API user session."""
        self.api_token = "test_token_" + str(random.randint(1000, 9999))
    
    @task(2)
    def analyze_repository(self):
        """Simulate repository analysis request."""
        payload = {
            "repository": "test-org/test-repo",
            "ref": "main",
            "options": {
                "include_security": True,
                "include_performance": True,
                "languages": ["python", "javascript"]
            }
        }
        
        headers = {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json"
        }
        
        with self.client.post(
            "/api/v1/analyze",
            json=payload,
            headers=headers,
            catch_response=True
        ) as response:
            if response.status_code in [200, 202]:  # Accept both sync and async
                response.success()
            else:
                response.failure(f"Analysis failed: {response.status_code}")
    
    @task(1)
    def get_analysis_status(self):
        """Check analysis status."""
        analysis_id = f"analysis_{random.randint(1000, 9999)}"
        
        headers = {
            "Authorization": f"Bearer {self.api_token}"
        }
        
        with self.client.get(
            f"/api/v1/analysis/{analysis_id}/status",
            headers=headers,
            catch_response=True
        ) as response:
            if response.status_code in [200, 404]:  # 404 for non-existent analysis
                response.success()
            else:
                response.failure(f"Status check failed: {response.status_code}")
    
    @task(1)
    def list_recent_analyses(self):
        """List recent analyses."""
        headers = {
            "Authorization": f"Bearer {self.api_token}"
        }
        
        params = {
            "limit": 10,
            "offset": 0
        }
        
        with self.client.get(
            "/api/v1/analyses",
            headers=headers,
            params=params,
            catch_response=True
        ) as response:
            if response.status_code in [200, 404]:
                response.success()
            else:
                response.failure(f"List analyses failed: {response.status_code}")


class StressTestUser(HttpUser):
    """High-frequency user for stress testing."""
    
    wait_time = between(0.1, 0.5)  # Very short wait times
    
    @task
    def rapid_health_checks(self):
        """Perform rapid health checks."""
        with self.client.get("/health", catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Health check failed: {response.status_code}")


# Custom events and listeners for detailed metrics
@events.request.add_listener
def request_handler(request_type, name, response_time, response_length, response, 
                   context, exception, start_time, url, **kwargs):
    """Custom request handler for detailed logging."""
    if exception:
        print(f"Request failed: {exception}")
    elif response_time > 5000:  # Log slow requests (>5s)
        print(f"Slow request: {name} took {response_time}ms")


@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    """Called when test starts."""
    print("Starting load test...")
    print(f"Target host: {environment.host}")


@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    """Called when test stops."""
    print("Load test completed.")
    
    # Print summary statistics
    stats = environment.stats
    print(f"Total requests: {stats.total.num_requests}")
    print(f"Total failures: {stats.total.num_failures}")
    print(f"Average response time: {stats.total.avg_response_time:.2f}ms")
    print(f"Max response time: {stats.total.max_response_time}ms")


# Load test scenarios
class LightLoadTest(HttpUser):
    """Light load test scenario."""
    wait_time = between(5, 15)
    weight = 3
    tasks = [WebhookUser.webhook_pr_opened, WebhookUser.health_check]


class MediumLoadTest(HttpUser):
    """Medium load test scenario."""
    wait_time = between(2, 8)
    weight = 2
    tasks = [
        WebhookUser.webhook_pr_opened,
        WebhookUser.webhook_pr_synchronize,
        APIUser.analyze_repository
    ]


class HeavyLoadTest(HttpUser):
    """Heavy load test scenario."""
    wait_time = between(0.5, 3)
    weight = 1
    tasks = [
        WebhookUser.webhook_pr_opened,
        WebhookUser.webhook_pr_synchronize,
        APIUser.analyze_repository,
        StressTestUser.rapid_health_checks
    ]


# Custom shapes for gradual load increase
from locust import LoadTestShape

class StepLoadShape(LoadTestShape):
    """
    Step load pattern that gradually increases load.
    """
    
    step_time = 60  # 60 seconds per step
    step_load = 10  # 10 users per step
    spawn_rate = 2  # 2 users per second
    time_limit = 600  # 10 minutes total
    
    def tick(self):
        run_time = self.get_run_time()
        
        if run_time > self.time_limit:
            return None
        
        current_step = run_time // self.step_time
        return (current_step * self.step_load, self.spawn_rate)


class SpikeLoadShape(LoadTestShape):
    """
    Spike load pattern with sudden increases.
    """
    
    def tick(self):
        run_time = self.get_run_time()
        
        if run_time < 60:
            return (5, 1)
        elif run_time < 120:
            return (50, 5)  # Sudden spike
        elif run_time < 180:
            return (10, 2)  # Back to normal
        elif run_time < 240:
            return (100, 10)  # Another spike
        else:
            return None