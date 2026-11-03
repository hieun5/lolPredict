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

### Prerequisites

- Python 3
- Node.js and npm
- Git
- Historical professional League of Legends match data
