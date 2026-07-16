# SuperKart Sales Prediction — Model Deployment

Deploys a trained sales prediction model (tuned XGBoost) as a containerized, decoupled web application, rather than leaving it stuck in a notebook. A Flask API serves predictions, and a separate Streamlit app provides an interactive frontend — each independently deployable and scalable.

## Architecture

```
┌──────────────────┐         ┌──────────────────┐
│  Streamlit UI     │  HTTP   │   Flask API       │
│  (frontend_files) │ ──────► │  (backend_files)  │
│  Docker container │         │  Docker container │
└──────────────────┘         └──────────────────┘
                                     │
                                     ▼
                          Serialized XGBoost model
                          (xgb_tuned_model.joblib)
```

- **Backend (Flask)**: loads the serialized model and exposes a `/v1/predict` endpoint that accepts product and store features as JSON and returns a predicted sales value, plus a `/health` check endpoint.
- **Frontend (Streamlit)**: form-based UI for entering product/store details and getting an instant sales prediction, calling the backend API over HTTP.
- **Containerization**: each side has its own `Dockerfile` and `requirements.txt`, deployed as separate Hugging Face Docker Spaces — meaning either component can be updated, redeployed, or scaled without touching the other.

## Why decoupled?

- **Independent scaling** — the backend can scale to handle prediction load without the frontend needing to change at all
- **Technology flexibility** — frontend and backend can each use whatever stack fits best (here, Streamlit + Flask)
- **Reusability** — the backend API can be called by other tools or integrated elsewhere, not just the one frontend

## Repo structure

```
superkart_deployment/
├── superkart_deployment.ipynb      (full orchestration notebook)
├── superkart_deployment.py         (orchestration code as a plain script)
├── backend_files/
│   ├── Backend_app.py              (Flask API)
│   ├── Backend_requirements.txt
│   └── Backend_Dockerfile
├── frontend_files/
│   ├── Frontend_app.py             (Streamlit UI)
│   ├── Frontend_requirements.txt
│   └── Frontend_Dockerfile
└── README.md
```

> **Note:** files here are prefixed `Backend_` / `Frontend_` for clarity when browsing the repo. Docker and Hugging Face Spaces expect the unprefixed names (`app.py`, `requirements.txt`, `Dockerfile`) — rename them back before using this as a deployable source.

## Notes

Replace the `[huggingface-username]` placeholder in the notebook and frontend app with your own Hugging Face Space details before running or redeploying.

## Tech stack

Python · Flask · Streamlit · XGBoost · Docker · Hugging Face Spaces · Gunicorn
