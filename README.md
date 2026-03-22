# ML Project Demo

A structured machine learning project template for rapid prototyping and experimentation.

## Project Structure

```
MLProjectDemo/
├── data/
│   ├── raw/              # Original, immutable data
│   └── processed/        # Cleaned and transformed data
├── models/               # Trained model artifacts
├── notebooks/            # Jupyter notebooks for exploration
├── src/
│   ├── __init__.py
│   ├── data/             # Data loading and preprocessing
│   │   ├── __init__.py
│   │   └── preprocess.py
│   ├── features/         # Feature engineering
│   │   ├── __init__.py
│   │   └── build_features.py
│   ├── models/           # Model training and evaluation
│   │   ├── __init__.py
│   │   ├── train.py
│   │   └── evaluate.py
│   └── utils/            # Utility functions
│       ├── __init__.py
│       └── helpers.py
├── tests/                # Unit tests
│   └── __init__.py
├── .gitignore
├── README.md
└── setup.py
```

## Setup

```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install the package in development mode
pip install -e ".[dev]"
```

## Usage

```python
from src.data.preprocess import load_data
from src.models.train import train_model

# Load and preprocess data
df = load_data("data/raw/dataset.csv")

# Train model
model = train_model(df)
```

## Running Tests

```bash
pytest tests/
```
