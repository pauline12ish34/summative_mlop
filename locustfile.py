"""
Locust Load Testing for Shoe Classification API
Simulates flood of requests to test API performance and scalability
"""

import os
import random
import time
from locust import HttpUser, task, between, events
from io import BytesIO
from PIL import Image


class ShoeClassifierUser(HttpUser):
    """
    Simulates a user making requests to the Shoe Classification API
    """
    
    # Wait between 1 and 3 seconds between tasks
    wait_time = between(1, 3)
    
    def on_start(self):
        """Called when a user starts"""
        self.test_image = self.create_test_image()
    
    def create_test_image(self):
        """Create a test image for prediction"""
        # Create a simple test image (128x128 to match model)
        img = Image.new('RGB', (128, 128), color=(random.randint(0, 255), 
                                                   random.randint(0, 255), 
                                                   random.randint(0, 255)))
        img_bytes = BytesIO()
        img.save(img_bytes, format='JPEG')
        img_bytes.seek(0)
        return img_bytes.getvalue()
    
    @task(5)
    def predict_shoe(self):
        """
        Make a prediction request (weight: 5)
        This is the most common operation
        """
        # Create file for upload
        files = {'file': ('test_shoe.jpg', BytesIO(self.test_image), 'image/jpeg')}
        
        with self.client.post("/predict", files=files, catch_response=True) as response:
            if response.status_code == 200:
                result = response.json()
                if 'predicted_class' in result:
                    response.success()
                else:
                    response.failure("Invalid response format")
            else:
                response.failure(f"Got status code {response.status_code}")
    
    @task(2)
    def check_status(self):
        """
        Check API status (weight: 2)
        Less frequent than predictions
        """
        with self.client.get("/status", catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Got status code {response.status_code}")
    
    @task(1)
    def get_metrics(self):
        """
        Get model metrics (weight: 1)
        Least frequent operation
        """
        with self.client.get("/metrics", catch_response=True) as response:
            if response.status_code == 200 or response.status_code == 404:
                # 404 is acceptable if no metrics available yet
                response.success()
            else:
                response.failure(f"Got status code {response.status_code}")
    
    @task(1)
    def health_check(self):
        """
        Health check endpoint (weight: 1)
        """
        with self.client.get("/health", catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Got status code {response.status_code}")


# Custom event handlers for detailed reporting
@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    """Called when the test starts"""
    print("\n" + "="*60)
    print("LOAD TEST STARTED")
    print("="*60)
    print(f"Target URL: {environment.host}")
    print(f"Users: {environment.runner.target_user_count if hasattr(environment.runner, 'target_user_count') else 'N/A'}")
    print("="*60 + "\n")


@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    """Called when the test stops"""
    print("\n" + "="*60)
    print("LOAD TEST COMPLETED")
    print("="*60)
    
    stats = environment.stats
    
    print("\n📊 REQUEST STATISTICS:")
    print("-" * 60)
    for name, stat in stats.entries.items():
        print(f"\nEndpoint: {name}")
        print(f"  Total Requests: {stat.num_requests}")
        print(f"  Failures: {stat.num_failures}")
        print(f"  Avg Response Time: {stat.avg_response_time:.2f}ms")
        print(f"  Min Response Time: {stat.min_response_time:.2f}ms")
        print(f"  Max Response Time: {stat.max_response_time:.2f}ms")
        print(f"  Requests/sec: {stat.total_rps:.2f}")
    
    print("\n" + "="*60)
    print(f"Total Requests: {stats.total.num_requests}")
    print(f"Total Failures: {stats.total.num_failures}")
    print(f"Average Response Time: {stats.total.avg_response_time:.2f}ms")
    print(f"Requests/sec: {stats.total.total_rps:.2f}")
    print("="*60 + "\n")


# For running with different scenarios
class QuickUser(HttpUser):
    """Quick user for rapid-fire requests"""
    wait_time = between(0.1, 0.5)
    
    @task
    def quick_predict(self):
        img = Image.new('RGB', (128, 128), color=(100, 150, 200))
        img_bytes = BytesIO()
        img.save(img_bytes, format='JPEG')
        img_bytes.seek(0)
        
        files = {'file': ('test.jpg', img_bytes, 'image/jpeg')}
        self.client.post("/predict", files=files)


class SlowUser(HttpUser):
    """Slow user simulating slower connections"""
    wait_time = between(5, 10)
    
    @task
    def slow_predict(self):
        img = Image.new('RGB', (128, 128), color=(200, 100, 50))
        img_bytes = BytesIO()
        img.save(img_bytes, format='JPEG')
        img_bytes.seek(0)
        
        files = {'file': ('test.jpg', img_bytes, 'image/jpeg')}
        self.client.post("/predict", files=files)


if __name__ == "__main__":
    # Run locust from command line
    print("Load testing script ready!")
    print("\nRun with:")
    print("  locust -f locustfile.py --host=http://localhost:8000")
    print("\n Or for headless mode:")
    print("  locust -f locustfile.py --host=http://localhost:8000 --users 100 --spawn-rate 10 --run-time 60s --headless")
