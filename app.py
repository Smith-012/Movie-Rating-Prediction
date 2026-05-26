"""
Streamlit Interactive App - Movie Rating Prediction
Shows model predictions with SHAP explainability
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
from pathlib import Path

# Page config
st.set_page_config(
    page_title="Movie Rating Predictor",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header { font-size: 3em; color: #ff7f0e; font-weight: bold; text-align: center; margin-bottom: 0.5em; }
    .metric-card { background-color: #f0f2f6; padding: 1.5em; border-radius: 0.5em; border-left: 4px solid #ff7f0e; }
    .prediction-box { background-color: #fff4e6; padding: 2em; border-radius: 0.5em; text-align: center; }
    .rating-display { color: #ff7f0e; font-weight: bold; font-size: 2.5em; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">🎬 Movie Rating Predictor</div>', unsafe_allow_html=True)

# Load model and preprocessor
@st.cache_resource
def load_model_and_data():
    model = joblib.load("artifacts/model.joblib")
    df_train = pd.read_csv("data/movies.csv")
    return model, df_train

try:
    model, df_train = load_model_and_data()
except FileNotFoundError:
    st.error("⚠️ Model not found! Please run `python main.py --data data/movies.csv --out artifacts` first.")
    st.stop()

# Sidebar - Input features
st.sidebar.header("🎥 Movie Details")

col1, col2 = st.sidebar.columns(2)
with col1:
    budget = st.number_input("Budget ($M)", min_value=0.0, max_value=300.0, value=100.0)
    runtime = st.number_input("Runtime (minutes)", min_value=50, max_value=250, value=120)
    release_year = st.number_input("Release Year", min_value=1900, max_value=2025, value=2020)

with col2:
    num_voted_users = st.number_input("Number of Voters", min_value=100, max_value=1000000, value=50000)
    genres = st.multiselect("Genres", ["Action", "Comedy", "Drama", "Horror", "Romance", "Sci-Fi"], default=["Drama"])

# Create input dataframe with sample structure
input_data = pd.DataFrame({
    'budget': [budget],
    'runtime': [runtime],
    'release_year': [release_year],
    'num_voted_users': [num_voted_users]
})

# Add genre features (one-hot encoded)
all_genres = ["Action", "Comedy", "Drama", "Horror", "Romance", "Sci-Fi"]
for genre in all_genres:
    input_data[f'genre_{genre.lower()}'] = [1 if genre in genres else 0]

# Main prediction
st.markdown("---")
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("📊 Rating Prediction")
    
    try:
        # Make prediction
        if isinstance(model, dict):
            preprocessor = model.get('preprocessor')
            regressor = model.get('model')
        else:
            if hasattr(model, 'named_steps'):
                preprocessor = model.named_steps.get('preprocessor')
                regressor = model.named_steps.get('regressor')
            else:
                preprocessor = None
                regressor = model
        
        # Transform input
        if preprocessor:
            input_transformed = preprocessor.transform(input_data)
        else:
            input_transformed = input_data
        
        # Get prediction
        prediction = regressor.predict(input_transformed)[0]
        
        # Display prediction
        st.markdown(f'<div class="prediction-box"><div class="rating-display">⭐ {prediction:.1f}/10</div></div>', unsafe_allow_html=True)
        st.markdown(f"**Predicted Rating:** {prediction:.2f} out of 10")
        
        # Model metrics
        st.markdown("### Model Performance")
        metrics_file = Path("artifacts/metrics.txt")
        if metrics_file.exists():
            metrics_text = metrics_file.read_text()
            for line in metrics_text.split('\n')[:5]:
                if line.strip():
                    st.text(line)
    
    except Exception as e:
        st.error(f"❌ Prediction error: {str(e)}")

with col2:
    st.subheader("🔍 SHAP Explainability")
    
    try:
        # Create SHAP explainer
        explainer = shap.TreeExplainer(regressor) if hasattr(regressor, 'tree_') else shap.KernelExplainer(regressor.predict, input_transformed[:1])
        shap_values = explainer.shap_values(input_transformed)
        
        # Handle array format
        if isinstance(shap_values, np.ndarray):
            shap_vals = shap_values
        else:
            shap_vals = shap_values
        
        # Display feature importance
        fig, ax = plt.subplots(figsize=(10, 4))
        feature_names = preprocessor.get_feature_names_out() if preprocessor else input_data.columns.tolist()
        
        # Create waterfall plot data
        importance_indices = np.argsort(np.abs(shap_vals[0]))[-5:]
        top_features = [feature_names[i] if isinstance(feature_names, np.ndarray) else feature_names[i] for i in importance_indices]
        top_values = shap_vals[0][importance_indices]
        
        ax.barh(range(len(top_features)), top_values)
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features)
        ax.set_xlabel('SHAP Value (Impact on Rating)')
        ax.set_title('Top Features Influencing Rating')
        plt.tight_layout()
        st.pyplot(fig)
    
    except Exception as e:
        st.warning(f"SHAP visualization: {str(e)}")
        st.info("📊 Feature importance loading...")

# Dataset Info
st.markdown("---")
st.subheader("📈 Dataset Overview")
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Movies", len(df_train))
    st.metric("Average Rating", f"{df_train.iloc[:, -1].mean():.1f}/10" if len(df_train.columns) > 1 else "N/A")

with col2:
    st.metric("Budget Range", f"${df_train['budget'].min():.0f}M - ${df_train['budget'].max():.0f}M")
    st.metric("Avg Budget", f"${df_train['budget'].mean():.0f}M")

with col3:
    st.metric("Runtime Range", f"{df_train['runtime'].min():.0f} - {df_train['runtime'].max():.0f} min")
    st.metric("Avg Runtime", f"{df_train['runtime'].mean():.0f} min")

st.markdown("""
---
**📚 About This App:**
- Built with Streamlit for interactive predictions
- Uses SHAP for model explainability
- Predicts movie ratings based on features
- Model: Gradient Boosting Regressor with hyperparameter tuning
""")
