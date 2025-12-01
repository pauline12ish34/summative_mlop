# Shoe Classification ML Pipeline

## Project Overview
End-to-end Machine Learning pipeline for shoe classification using deep learning. The system classifies shoe images into three categories: **Boot**, **Sandal**, and **Shoe**.


youtube video: https://youtu.be/kgZqRPuDZy8


## Features
- ✅ Deep learning model with transfer learning (ResNet50)
- ✅ RESTful API for predictions
- ✅ Web UI dashboard with visualizations
- ✅ Model retraining capability
- ✅ Docker containerization
- ✅ Load testing with Locust
- ✅ Cloud deployment ready

## Directory Structure
```
pipeline_summative/
│
├── README.md
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── locustfile.py
│
├── notebook/
│   └── shoe_prediction_ml.ipynb      # Model development & evaluation
│
├── src/
│   ├── preprocessing.py              # Data preprocessing
│   ├── model.py                      # Model architecture & training
│   ├── prediction.py                 # Prediction logic
│   ├── app.py                        # FastAPI application
│   └── streamlit_app.py              # Web UI dashboard
│
├── data/
│   ├── train/                        # Training data
│   ├── test/                         # Test data
│   └── uploads/                      # User uploaded data
│
└── models/
    ├── shoe_classifier_model.h5      # Trained model
    └── class_names.json              # Class labels
```

## Setup Instructions

### Prerequisites
- Python 3.10+
- Docker (optional)
- Git

### 1. Clone Repository
```bash
git clone https://github.com/pauline12ish34/summative_mlop.git
cd pipeline_summative
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Download Dataset
The notebook automatically downloads the dataset from Kaggle using `kagglehub`.

### 4. Train Model
Run the Jupyter notebook to train the model:
```bash
jupyter notebook notebook/shoe_prediction_ml.ipynb
```
Execute all cells to:
- Download and explore data
- Preprocess images
- Train model with transfer learning
- Evaluate with multiple metrics
- Save model to `models/`

## Running the Application

### Option 1: Local Development

#### Start API Server
```bash
python src/app.py
```
API will be available at: `http://localhost:8000`

#### Start Web UI
```bash
streamlit run src/streamlit_app.py
```
Dashboard will be available at: `http://localhost:8501`

### Option 2: Docker

#### Build and Run
```bash
docker-compose up --build
```

Services:
- API: `http://localhost:8000`
- Web UI: `http://localhost:8501`

#### Scale API Containers
```bash
docker-compose up --scale api=3
```

## API Endpoints

### Prediction
- **POST** `/predict` - Classify shoe image
  ```bash
  curl -X POST -F "file=@shoe.jpg" http://localhost:8000/predict
  ```

### Data Upload
- **POST** `/upload-data` - Upload training data
  ```bash
  curl -X POST -F "files=@image1.jpg" -F "files=@image2.jpg" "http://localhost:8000/upload-data?category=Boot"
  ```

### Retraining
- **POST** `/retrain` - Trigger model retraining

### Monitoring
- **GET** `/status` - Get API status
- **GET** `/metrics` - Get model metrics
- **GET** `/health` - Health check

## Model Details

### Architecture
- **Base Model**: ResNet50 (pre-trained on ImageNet)
- **Custom Layers**: GlobalAveragePooling2D, BatchNormalization, Dense layers with Dropout

### Optimization Techniques
1. **Transfer Learning**: ResNet50 pre-trained weights
2. **Data Augmentation**: Rotation, shifts, zoom, flip
3. **Regularization**: Dropout (0.5, 0.3), BatchNormalization
4. **Early Stopping**: Patience=5 on validation loss
5. **Optimizer**: Adam (lr=0.001)

### Evaluation Metrics
- Accuracy
- Loss
- Precision
- Recall
- F1-Score
- AUC-ROC

## Load Testing

### Run Locust Tests
```bash
# With Web UI
locust -f locustfile.py --host=http://localhost:8000

# Headless mode
locust -f locustfile.py --host=http://localhost:8000 --users 100 --spawn-rate 10 --run-time 60s --headless
```

## Web UI Features
1. **Home Dashboard** - System overview
2. **Predict** - Upload and classify images
3. **Analytics** - Model performance metrics
4. **Upload Data** - Bulk image upload
5. **Retrain Model** - Trigger retraining
6. **System Status** - Monitor API health

## Retraining Process
1. Upload new images via Web UI or API
2. Trigger retraining
3. System preprocesses data
4. Fine-tunes existing model
5. Evaluates and saves updated model

## Cloud Deployment
 deployment was done to:
- Render

## Author
Pauline Ishimwe

## Repository
https://github.com/pauline12ish34/summative_mlop


