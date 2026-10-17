# LoL Pro Predictor

A full-stack machine learning application that predicts the outcomes of professional League of Legends matches using historical esports data, team performance statistics, and predictive analytics.

Built with React, TypeScript, FastAPI, and scikit-learn to provide data-driven match predictions and interactive team comparisons.

## Features

🎮 **Match Outcome Prediction**: Predicts professional League of Legends match winners using machine learning models trained on historical esports data.

📊 **Team Performance Analysis**: Analyzes team win rates, recent match performance, and historical results to identify competitive trends.

🤖 **Machine Learning Models**: Uses Logistic Regression and Random Forest classifiers to predict match outcomes and estimate win probabilities.

📈 **Feature Engineering**: Extracts meaningful performance metrics from historical match data to improve prediction accuracy.

⚔️ **Head-to-Head Comparisons**: Compares professional teams using historical statistics, recent performance, and previous matchups.

🏆 **Professional Esports Data**: Integrates historical professional League of Legends match datasets for model training and analysis.

💻 **Interactive Dashboard**: Displays match predictions, team statistics, and model-generated win probabilities through a modern web interface.

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/hieun5/lolPredictor.git
cd lolPredictor
```

### 2. Install backend dependencies

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 3. Start the backend

```powershell
python -m uvicorn main:app --reload
```

### 4. Install frontend dependencies

Open a second terminal:

```bash
cd frontend
npm install
```

### 5. Start the frontend

```bash
npm run dev
```

Open the local development URL displayed in the terminal.

## Architecture

- **Frontend**: React, TypeScript, Vite
- **Backend**: Python, FastAPI
- **Machine Learning**: scikit-learn, Logistic Regression, Random Forest
- **Data Processing**: pandas, NumPy
- **Data Visualization**: Recharts
- **Data Source**: Historical professional League of Legends esports datasets

## How It Works

1. **Data Collection** — Historical professional match results and team statistics are collected and organized.
2. **Data Preprocessing** — Raw datasets are cleaned, standardized, and prepared for machine learning.
3. **Feature Engineering** — Team win rates, recent performance, and historical matchup statistics are converted into predictive features.
4. **Model Training** — Logistic Regression and Random Forest models are trained on historical match outcomes.
5. **Model Evaluation** — Models are evaluated using held-out match data and classification metrics.
6. **Match Prediction** — Users select two professional teams to generate predicted win probabilities.
7. **Results Visualization** — The dashboard displays the predicted winner, probability estimates, and supporting team statistics.

## Getting Started

### Prerequisites

- Python 3
- Node.js and npm
- Git
- Historical professional League of Legends match data

## Project Purpose

LoL Pro Predictor demonstrates the application of machine learning, statistical analysis, and full-stack development to competitive esports. The project focuses on transforming historical match data into meaningful predictions through feature engineering, supervised learning, and interactive visualization.

## Author

GitHub: [hieun5](https://github.com/hieun5)
