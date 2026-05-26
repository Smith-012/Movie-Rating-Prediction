# Movie Rating Prediction

## Portfolio Summary
This project upgrades the movie rating baseline into a more complete regression workflow. It compares candidate regressors, tunes them with cross-validation, and persists the best artifact together with a metrics report.

## What Makes It Portfolio-Ready

- Model comparison between `RandomForestRegressor` and `GradientBoostingRegressor`
- Hyperparameter search with cross-validation
- Reusable preprocessing pipeline
- Saved model and metrics files for reproducibility

## 🎬 What This Project Does

This project predicts movie ratings (IMDb score) based on movie metadata using machine learning. Given movie characteristics (budget, runtime, release year, genres, etc.), the model estimates the expected audience rating on a scale of 1-10. It demonstrates:

- **Regression modeling**: Building models to predict continuous values (rating scores)
- **Feature interaction**: Understanding how budget and runtime influence ratings differently for different genres
- **Model comparison**: Training multiple regressors (Random Forest, Gradient Boosting) to minimize prediction error
- **Hyperparameter tuning**: Using GridSearchCV with R² as the optimization metric
- **Production inference**: Batch prediction with validation for new movie metadata
- **Interactive web app**: Real-time rating predictions with SHAP feature importance

### Problem Statement

Movie studios, streaming platforms, and investors want to estimate likely audience reception before or shortly after release. This project builds a predictive model that identifies which movie characteristics correlate with higher ratings. Understanding rating drivers helps studios make production decisions and marketing strategies.

### Key Features Used

- **Budget**: Production budget in millions (continuous)
- **Runtime**: Movie duration in minutes (continuous)
- **Release Year**: Year of release (temporal)
- **Genre**: Movie category - Action, Comedy, Drama, etc. (categorical, multi-label)
- **Language**: Original language (categorical)
- **Director Recognition**: Whether director has previous hit movies (derived feature)
- **Season**: Month of release - seasonal patterns (categorical)

### Models Trained

1. **Random Forest Regression**: Ensemble approach with non-linear relationships
2. **Gradient Boosting Regression**: Sequential boosting for improved R² scores
3. **Includes validation metrics**: MAE, RMSE, and R² to evaluate different aspects

## Tech Stack

- Python 3
- Pandas and NumPy
- Scikit-learn
- Joblib

## Dataset
Place the CSV file here:

- `data/movies.csv`

Target column can be one of:

- `rating`
- `Rating`
- `imdb_rating`
- `score`
- `user_rating`

## Installation

### Development
```bash
pip install -r requirements.txt
# or with pinned versions
pip install -r requirements-lock.txt
```

### Production (via pip)
```bash
pip install -e .
```

This registers the CLI command `train-movie` globally.

## Training

```bash
# Using the CLI (after install -e .)
train-movie --data data/movies.csv --out artifacts

# Or directly
python main.py --data data/movies.csv --out artifacts
```

Outputs:
- `artifacts/model.joblib` – trained regressor
- `artifacts/metrics.txt` – evaluation metrics (MAE, RMSE, R²)

## Inference

### Batch Prediction
```bash
python inference.py --model artifacts/model.joblib --input new_movies.csv --output predictions.json
```

### Programmatic Usage
```python
from inference import load_model, predict
import pandas as pd

model = load_model("artifacts/model.joblib")
df = pd.read_csv("new_movies.csv")
result = predict(model, df)
print(result)  # {"predictions": [...], "mse": ...}
```

## Interactive Web App (Streamlit)

### Launch the Web App
```bash
# Install Streamlit and SHAP first
pip install streamlit shap

# Run the app
streamlit run app.py
```

The app provides:
- **Interactive predictions**: Enter movie details and get predicted ratings
- **SHAP explanations**: Understand feature importance for rating predictions
- **Model metrics**: View MAE, RMSE, R² performance metrics
- **Dataset overview**: Explore movie rating patterns and trends

**Browser**: Opens automatically at `http://localhost:8501`

## Jupyter Notebooks

Explore the analysis and training workflow:

```bash
jupyter notebook notebooks/
```

**Available notebooks:**
- `01_eda.ipynb` - Exploratory Data Analysis (budget/runtime/ratings analysis, genre impact)
- Add more notebooks for detailed analysis

## Docker

```bash
# Build
docker build -t movie-rating:latest .

# Train in container
docker run --rm -v $(pwd)/data:/app/data -v $(pwd)/artifacts:/app/artifacts movie-rating:latest

# Predict
docker run --rm -v $(pwd):/app movie-rating:latest python inference.py --input /app/new_movies.csv
```

## Testing

```bash
python -m unittest discover -v tests
```

## Production Guide

### Prerequisites
- Python 3.11+
- Docker (optional, for containerized deployment)
- Git

### Local Setup

```bash
# Clone the repo
git clone <repo-url>
cd Portfolio-Task-2-Movie-Rating-Production-Style

# Install dependencies (locked versions recommended for production)
pip install -r requirements-lock.txt
# or
pip install -r requirements.txt

# Install the package
pip install -e .
```

### Production Monitoring

**Key Metrics:**
- **Regression Error**: Monitor MAE and RMSE (see artifacts/metrics.txt)
- **Input Validation**: Ensure all required features are present
- **Inference Latency**: Track prediction time
- **Prediction Range**: Monitor mean and std deviation of predicted ratings

**Alerts:**
- Set alert if MAE increases > 10% from baseline
- Alert if > 5% of requests fail validation
- Alert if average latency exceeds 50ms per sample

**Logging:**
All inference runs log validation and prediction stats:
```
2026-05-26 10:30:45 - INFO - Input validation passed: 500 rows, 15 columns
2026-05-26 10:30:46 - INFO - Loaded model from artifacts/model.joblib
2026-05-26 10:30:47 - INFO - Generated predictions for 500 samples
```

### CI/CD

GitHub Actions workflow at `.github/workflows/ci.yml` runs:
- Dependency installation
- Unit tests (`python -m unittest discover`)
- On every push/PR

### Model Retraining

To retrain with new data:
```bash
python main.py --data data/movies_updated.csv --out artifacts
```

New artifacts will overwrite previous `model.joblib` and `metrics.txt`.

### Maintenance

- Keep `requirements-lock.txt` synchronized with production environment
- Test any dependency upgrades in staging first
- Monitor feature distributions for data drift
- Archive old models before retraining

## Project Flow

1. Detect the target rating column.
2. Split features into numeric and categorical groups.
3. Impute missing values and encode categories.
4. Compare candidate regressors (Random Forest, Gradient Boosting) under cross-validation.
5. Save the best model and its test metrics.

## 📊 Limitations

### Current Constraints

1. **Metadata-Only Features**: The model uses only metadata (budget, runtime, genre) but not content-based features like plot quality, cinematography, or performances
2. **Limited Data Scope**: Assumes movie ratings follow the same patterns as the training dataset; may not generalize to niche or experimental films
3. **Missing Cultural Context**: Release date and location features are not included; cultural events and holidays influence ratings
4. **No Review Text Analysis**: Ignores actual user review text which contains rich sentiment information
5. **Temporal Bias**: Older movies in dataset may have fewer reviews or different rating patterns than newer releases
6. **Genre Simplification**: Movies are assigned to single/few genres; complex multi-genre relationships lost
7. **External Factors Ignored**: Marketing budget, critical reception, franchise history, and star power not captured

### Model Trade-offs

- **Speed vs. Accuracy**: Random Forest is faster but Gradient Boosting is more accurate
- **Interpretability Loss**: Tree-based ensemble models are harder to explain than linear regression
- **Tuning Complexity**: Many hyperparameters increase training time and risk of overfitting

## 🚀 Future Scope

### Short-term Enhancements (Weeks)

1. **Feature Engineering**: Extract director/actor reputation scores, franchise indicators, sequel status
2. **Release Season Analysis**: Add detailed seasonal patterns (summer blockbuster vs. prestige awards season)
3. **Ensemble Methods**: Combine Random Forest + Gradient Boosting with voting/stacking
4. **Error Analysis Notebook**: Add `02_error_analysis.ipynb` to understand prediction mistakes
5. **Prediction Confidence**: Add uncertainty quantification for low-confidence predictions

### Medium-term Extensions (Months)

1. **NLP Integration**: Process plot summaries and user reviews using TF-IDF or embeddings
2. **Time-Series Modeling**: Capture rating evolution post-release (early ratings vs. long-term average)
3. **Critic vs. Audience Split**: Model ratings separately for critics and general audience
4. **API Development**: Build FastAPI endpoint for real-time rating predictions
5. **Dashboard**: Create Streamlit dashboard comparing actual vs. predicted ratings with trends
6. **Recommendation System**: Extend to suggest movies similar to top-predicted ratings

### Long-term Vision (Years)

1. **Deep Learning**: Use neural networks to learn complex feature interactions automatically
2. **Multimodal Learning**: Incorporate movie posters, trailers, and soundtracks as inputs
3. **Causal Discovery**: Understand which factors truly influence ratings vs. correlations
4. **User-Specific Predictions**: Predict ratings for individual users based on their preferences
5. **Genre Transfer Learning**: Fine-tune on genre-specific rating patterns (blockbusters vs. indie films)
6. **Fairness Analysis**: Ensure model doesn't systematically underrate/overrate movies by gender/nationality/language

### Research Opportunities

- Investigate relationship between production budget and audience satisfaction
- Analyze whether high-budget movies tend to have polarized ratings (love it or hate it)
- Compare rating patterns across different streaming platforms (Netflix, Amazon, etc.)
- Study impact of director/actor changes on sequel ratings

## Project Structure

```
.
├── main.py              # Training script
├── inference.py         # Inference with validation
├── app.py               # Streamlit interactive web app
├── pyproject.toml       # Package metadata
├── Dockerfile           # Container definition
├── Makefile             # Convenience commands
├── requirements.txt     # Python dependencies
├── requirements-lock.txt # Pinned versions
├── LICENSE              # MIT
├── README.md            # This file
├── .gitignore           # Git ignore patterns
├── .github/workflows/ci.yml # GitHub Actions
├── data/
│   └── movies.csv       # Training dataset
├── notebooks/           # Jupyter notebooks
│   └── 01_eda.ipynb     # Exploratory Data Analysis
├── artifacts/           # Trained model & metrics
└── tests/
    └── test_quick.py    # Smoke tests
```

## Tech Stack

- **Python 3.11+**
- **Pandas & NumPy** – data manipulation
- **Scikit-learn** – ML models and preprocessing
- **Joblib** – model persistence
- **Matplotlib & Seaborn** – visualization
