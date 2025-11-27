# Deployment Guide

## Cloud Deployment Options

### Option 1: Google Cloud Platform (Recommended)

#### Prerequisites
- GCP account with billing enabled
- Google Cloud SDK installed
- Docker installed

#### Steps

1. **Set up GCP project**
```bash
gcloud init
gcloud config set project YOUR_PROJECT_ID
```

2. **Enable required APIs**
```bash
gcloud services enable cloudbuild.googleapis.com
gcloud services enable run.googleapis.com
gcloud services enable containerregistry.googleapis.com
```

3. **Build and push Docker image**
```bash
# Build
gcloud builds submit --tag gcr.io/YOUR_PROJECT_ID/shoe-classifier

# Or use Docker
docker build -t gcr.io/YOUR_PROJECT_ID/shoe-classifier .
docker push gcr.io/YOUR_PROJECT_ID/shoe-classifier
```

4. **Deploy to Cloud Run**
```bash
gcloud run deploy shoe-classifier \
  --image gcr.io/YOUR_PROJECT_ID/shoe-classifier \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 4Gi \
  --cpu 2 \
  --timeout 300 \
  --max-instances 10
```

5. **Get the URL**
```bash
gcloud run services describe shoe-classifier --region us-central1
```

#### Scaling Configuration
```bash
# Set min/max instances
gcloud run services update shoe-classifier \
  --min-instances 1 \
  --max-instances 10 \
  --region us-central1

# Set concurrency
gcloud run services update shoe-classifier \
  --concurrency 80 \
  --region us-central1
```

---

### Option 2: AWS (Elastic Container Service)

#### Prerequisites
- AWS account
- AWS CLI installed
- Docker installed

#### Steps

1. **Create ECR repository**
```bash
aws ecr create-repository --repository-name shoe-classifier
```

2. **Login to ECR**
```bash
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin YOUR_ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com
```

3. **Build and push**
```bash
docker build -t shoe-classifier .
docker tag shoe-classifier:latest YOUR_ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/shoe-classifier:latest
docker push YOUR_ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/shoe-classifier:latest
```

4. **Create ECS cluster**
```bash
aws ecs create-cluster --cluster-name shoe-classifier-cluster
```

5. **Create task definition**
Create `task-definition.json`:
```json
{
  "family": "shoe-classifier",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "2048",
  "memory": "4096",
  "containerDefinitions": [
    {
      "name": "shoe-classifier",
      "image": "YOUR_ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/shoe-classifier:latest",
      "portMappings": [
        {
          "containerPort": 8000,
          "protocol": "tcp"
        }
      ]
    }
  ]
}
```

Register task:
```bash
aws ecs register-task-definition --cli-input-json file://task-definition.json
```

6. **Create service**
```bash
aws ecs create-service \
  --cluster shoe-classifier-cluster \
  --service-name shoe-classifier-service \
  --task-definition shoe-classifier \
  --desired-count 2 \
  --launch-type FARGATE
```

---

### Option 3: Render (Easiest)

1. **Create account** at [render.com](https://render.com)

2. **Connect GitHub repository**

3. **Create new Web Service**
   - Select repository: `summative_mlop`
   - Name: `shoe-classifier`
   - Environment: `Docker`
   - Region: Choose closest
   - Instance Type: Standard (2GB RAM minimum)

4. **Environment variables** (if needed)
   ```
   PYTHONUNBUFFERED=1
   ```

5. **Deploy** - Render will automatically build and deploy

6. **Custom Domain** (optional)
   - Go to Settings
   - Add custom domain
   - Update DNS records

---

### Option 4: Heroku

1. **Install Heroku CLI**
```bash
heroku login
```

2. **Create app**
```bash
heroku create shoe-classifier-app
```

3. **Set stack to container**
```bash
heroku stack:set container -a shoe-classifier-app
```

4. **Deploy**
```bash
git push heroku main
```

5. **Scale**
```bash
heroku ps:scale web=2 -a shoe-classifier-app
```

---

## Load Balancing & Scaling

### Docker Compose (Local/VPS)
```bash
# Scale to 3 instances
docker-compose up --scale api=3
```

### Kubernetes
Create `deployment.yaml`:
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: shoe-classifier
spec:
  replicas: 3
  selector:
    matchLabels:
      app: shoe-classifier
  template:
    metadata:
      labels:
        app: shoe-classifier
    spec:
      containers:
      - name: shoe-classifier
        image: gcr.io/YOUR_PROJECT_ID/shoe-classifier
        ports:
        - containerPort: 8000
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
---
apiVersion: v1
kind: Service
metadata:
  name: shoe-classifier-service
spec:
  selector:
    app: shoe-classifier
  ports:
  - port: 80
    targetPort: 8000
  type: LoadBalancer
```

Deploy:
```bash
kubectl apply -f deployment.yaml
```

---

## Monitoring & Logging

### Prometheus + Grafana
Add to `docker-compose.yml`:
```yaml
prometheus:
  image: prom/prometheus
  ports:
    - "9090:9090"
  volumes:
    - ./prometheus.yml:/etc/prometheus/prometheus.yml

grafana:
  image: grafana/grafana
  ports:
    - "3000:3000"
```

### Cloud Logging
- **GCP**: Automatically available in Cloud Logging
- **AWS**: Use CloudWatch
- **Azure**: Use Application Insights

---

## Performance Optimization

### 1. Model Optimization
```python
# Convert to TensorFlow Lite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
```

### 2. Caching
Add Redis for prediction caching:
```yaml
redis:
  image: redis:alpine
  ports:
    - "6379:6379"
```

### 3. CDN for Static Assets
- Use CloudFlare
- AWS CloudFront
- Google Cloud CDN

### 4. Database for Metadata
Add PostgreSQL for storing predictions:
```yaml
postgres:
  image: postgres:13
  environment:
    POSTGRES_PASSWORD: password
  ports:
    - "5432:5432"
```

---

## Security Best Practices

1. **API Authentication**
```python
from fastapi.security import HTTPBearer

security = HTTPBearer()

@app.post("/predict")
async def predict(file: UploadFile, token: str = Depends(security)):
    # Verify token
    pass
```

2. **HTTPS Only**
```python
# Force HTTPS
@app.middleware("http")
async def force_https(request, call_next):
    if request.url.scheme != "https":
        url = request.url.replace(scheme="https")
        return RedirectResponse(url)
    return await call_next(request)
```

3. **Rate Limiting**
```bash
pip install slowapi
```

4. **Environment Variables**
```bash
# Never commit secrets
export API_KEY="your-secret-key"
export DB_PASSWORD="password"
```

---

## Cost Optimization

### GCP
- Use Cloud Run (pay per use)
- Set min instances to 0
- Use preemptible VMs for batch jobs

### AWS
- Use spot instances
- Enable auto-scaling
- Use S3 for model storage

### General
- Compress images before processing
- Cache predictions
- Use model quantization
- Implement request batching

---

## Troubleshooting

### Issue: Out of Memory
**Solution**: Increase container memory or use model quantization

### Issue: Slow predictions
**Solution**: Enable GPU, use TensorFlow Lite, batch requests

### Issue: Container fails to start
**Solution**: Check logs, verify dependencies, ensure model file exists

### Issue: High latency
**Solution**: Add caching, use CDN, scale horizontally

---

## Support

For deployment issues:
1. Check logs
2. Verify environment variables
3. Test locally with Docker
4. Contact cloud provider support

## Additional Resources
- [FastAPI Deployment](https://fastapi.tiangolo.com/deployment/)
- [Docker Best Practices](https://docs.docker.com/develop/dev-best-practices/)
- [Kubernetes Tutorial](https://kubernetes.io/docs/tutorials/)
