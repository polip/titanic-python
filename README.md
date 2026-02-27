# Titanic Survival Predictor - Python Version

This is a Python translation of the original R-based Titanic survival prediction project. It includes machine learning model training, web applications, and deployment configurations.

## Project Structure

```
python_version/
├── clean_data.py          # Data preprocessing (translation of cleaning.R)
├── train_model.py         # ML model training (translation of titanic_training.R)  
├── streamlit_app.py          # Streamlit web app (translation of app.R)
├── titanic_fastapi.py       # REST API (translation of plumber.R)
├── run_pipeline.py          # Pipeline runner script
├── requirements.txt         # Python dependencies
├── Dockerfile.streamlit     # Docker for Streamlit app
├── Dockerfile.fastapi-gcs       # Docker for FastAPI app
├── docker-compose.yml       # Multi-container deployment
└── README.md               # This file
```

## Quick Start

### Option 1: Automated Pipeline

Run the complete pipeline with one command:

```bash
cd python_version
python run_pipeline.py --start-streamlit
```
Without options script just clean data and train model.

Options:
- `--skip-training`: Skip model training (use existing model)
- `--start-streamlit`: Start Streamlit app after training
- `--start-fastapi`: Start FastAPI app after training  
- `--docker`: Use Docker containers
- `--deploy`: Deploy FastAPI to Google Cloud Run

### Option 2: Manual Steps

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

2. **Run data cleaning:**
```bash
python clean_data.py
```

3. **Train the model:**
```bash
python train_model.py
```

4. **Start web applications:**

   **Streamlit (Interactive UI):**
   ```bash
   streamlit run streamlit_app.py
   ```
   Access at: http://localhost:8501

   **FastAPI (REST API):**
   ```bash
   uvicorn titanic_fastapi:app --host 0.0.0.0 --port 8000
   ```
   Access at: http://localhost:8000
   API docs at: http://localhost:8000/docs

5. **Build and deploy docker API to Google Cloud Run:**
```bash
python deploy_api_gcs.py
```

### Option 3: Docker Deployment Local

```bash
docker-compose up --build
```

This starts both applications:
- Streamlit: http://localhost:8501
- FastAPI: http://localhost:8000

## Features

### Data Processing (`clean_data.py`)
- Converts R data cleaning logic to pandas
- Feature engineering: title extraction from names
- Categorical encoding and missing value handling

### Machine Learning (`train_model.py`)
- Random Forest classifier (equivalent to R's ranger)
- Preprocessing pipeline with KNN imputation and scaling
- Feature importance analysis and visualization
- Cross-validation evaluation
- Test set predictions and submission file generation

### Web Applications

**Streamlit App (`streamlit_app.py`)**
- Interactive web interface (translation of Shiny app)
- Single passenger prediction mode
- CSV batch upload functionality
- Interactive visualizations with Plotly
- Bootstrap-style theming

**FastAPI App (`titanic_fastapi.py`)**
- REST API endpoints (translation of plumber.R)
- Single prediction: `POST /predict`
- Batch prediction: `POST /predict/batch`
- Custom JSON input: `POST /predict/custom`
- Automatic API documentation

### Deployment - local
- **Docker**: Separate containers for each app
- **Docker Compose**: Multi-container orchestration
- **Health checks**: Built-in application monitoring
- **Security**: Non-root user execution

## API Usage Examples

### Single Prediction
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "pclass": "1st",
    "sex": "female", 
    "age": 25,
    "sibsp": 0,
    "parch": 0,
    "fare": 50.0,
    "embarked": "S",
    "title": "Miss"
  }'
```

### Batch Prediction
```bash
curl -X POST "http://localhost:8000/predict/batch" \
  -F "file=@passengers.csv"
```

## Model Performance

The Python version achieves similar performance to the original R implementation:
- Random Forest classifier with 100 trees
- Cross-validation accuracy: ~82%
- Feature importance analysis available
- Handles missing values with KNN imputation

## Differences from R Version

| Aspect | R Version | Python Version |
|--------|-----------|----------------|
| **Data Processing** | tidyverse, janitor | pandas |
| **ML Framework** | tidymodels, ranger | scikit-learn |
| **Web UI** | Shiny | Streamlit |
| **API** | plumber | FastAPI |
| **Deployment** | vetiver, Docker | Docker, docker-compose |
| **Visualization** | ggplot2 | matplotlib, plotly |

## Requirements

- Python 3.12+
- See `requirements.txt` for full dependency list
- Docker (optional, for containerized deployment)

## Troubleshooting

1. **Module import errors**: Install requirements with `pip install -r requirements.txt`
2. **Model not found**: Run `python model_training.py` first
3. **Port conflicts**: Change ports in docker-compose.yml or command line arguments
4. **Data not found**: Ensure train.csv and test.csv are in the data/ directory

## Next Steps

1. **Model Improvements**: Hyperparameter tuning, feature engineering
2. **Monitoring**: Add logging, metrics, and alerting
3. **Testing**: Unit tests and integration tests
4. **CI/CD**: Automated testing and deployment pipelines