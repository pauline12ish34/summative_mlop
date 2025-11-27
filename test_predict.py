"""Test prediction endpoint"""
import requests

# Get a test image
image_path = r"demo_upload\Boot\00000003.jpg"

with open(image_path, 'rb') as f:
    files = {'file': ('test.jpg', f, 'image/jpeg')}
    response = requests.post('http://localhost:8000/predict', files=files)
    
print(f"Status Code: {response.status_code}")
print(f"Response: {response.text}")
