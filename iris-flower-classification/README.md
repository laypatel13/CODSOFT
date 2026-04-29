# Iris Flower Classification

A machine learning project that classifies Iris flowers into three species —
Setosa, Versicolor, and Virginica — based on sepal and petal measurements.

Built as part of the **CodSoft Data Science Internship**.

## Dataset
- 150 samples, 4 features, 3 classes
- Features: sepal_length, sepal_width, petal_length, petal_width
- Classes: Iris-setosa, Iris-versicolor, Iris-virginica

## Project Structure

```
iris-flower-classification/
├── IRIS.csv                  # Dataset
├── explore.py                # EDA - stats, pairplot, heatmap
├── train.py                  # Model training & evaluation
├── main.py                   # Full pipeline entry point
├── test_iris.py              # Pytest tests
├── requirements.txt          # Dependencies
└── .gitignore
```

## Setup
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Run
```bash
# Full pipeline
python main.py

# Tests only
pytest test_iris.py -v
```

## Model
- Algorithm: Random Forest Classifier
- Train/Test Split: 80/20
- Accuracy: ~97%

## Results
- High accuracy across all three species
- Confusion matrix, pairplot, and heatmap generated on run

## Tools & Libraries
Python | pandas | numpy | scikit-learn | matplotlib | seaborn | pytest