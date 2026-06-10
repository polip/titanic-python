# titanic-python — Agent Guide

## Project
Titanic survival prediction: R -> Python translation. Random Forest classifier, Streamlit UI + FastAPI REST API.

## Key files

| File | Purpose |
|---|---|
| `clean_data.py` | Loads `data/train.csv`, cleans/encodes, saves `data/train_clean.pkl` |
| `train_model.py` | Trains RF pipeline (KNN impute → scale → OrdinalEncode → RF), outputs `models/titanic_model_ct.pkl` |
| `streamlit_app.py` | Streamlit UI on `:8501`. Requires model + features in `models/` |
| `titanic_fastapi.py` | FastAPI on `:8000`. Loads model: GCS → local `model/` → dummy fallback |
| `utils.py` | Shared encoding maps (`PCLASS_MAP`, `SEX_MAP`, `EMBARKED_MAP`, `TITLE_MAP`) |
| `run_pipeline.py` | Orchestrator: clean → train → (optional) streamlit/fastapi/docker/deploy |
| `deploy_api_gcs.py` | Deploys FastAPI to Cloud Run via gcloud CLI |

## Commands

```bash
# Pipeline (standard)
python clean_data.py && python train_model.py

# With MLflow tracking (auto-detected; tracking URI: sqlite:///mlflow.db)
python train_model.py

# Apps
streamlit run streamlit_app.py              # http://localhost:8501
uvicorn titanic_fastapi:app --host 0.0.0.0 --port 8000  # http://localhost:8000/docs

# Orchestrated
python run_pipeline.py --start-streamlit    # full pipeline + streamlit
python run_pipeline.py --docker             # docker-compose up --build

# Deploy to Cloud Run
python deploy_api_gcs.py                    # requires gcloud + docker + auth
```

## Environment

- `.env` (dotenv format) — local development
- `.env.gcs` (JSON, `--env-vars-file` format) — Cloud Run
- Key vars: `GOOGLE_CLOUD_PROJECT`, `GOOGLE_CLOUD_REGION`, `BUCKET_NAME`, `MODEL_FILE`, `FEATURES_FILE`, `LOCAL_MODEL_DIR`

## Package management

- `uv.lock` present — project uses **uv** (`uv add`, `uv run`)
- `pip install -r requirements.txt` also works
- Python 3.12 (`.python-version`)

## Architecture notes

- **Pipeline order matters**: `clean_data.py` → `train_model.py` → apps (the pipeline reads `data/train_clean.pkl`)
- **Model loading priority** (FastAPI): GCS bucket → `model/` local dir → dummy sklearn fallback
- **Two model directories**: `models/` holds training output, `model/` is the FastAPI local-fallback dir
- **Data + models gitignored** — `data/` and `models/` are in `.gitignore` but contain tracked files (CSVs, pickles, plots)
- **No test suite** configured
- **No formatter/linter config** — `black`, `isort` listed as deps but no config or script; `.ruff_cache/` in gitignore but no ruff config

## Deployment

- GitHub Actions (`deploy-cloud-run.yml`): triggers on `main` pushes touching app/deploy files
- Manual: `python deploy_api_gcs.py [deploy|upload-models|url|check]`
- Docker Compose serves both apps behind bridge network `titanic-network`
