# Housing Price Prediction – ML Ops Assignment

This repository contains a complete machine learning workflow for predicting housing prices using the Boston Housing dataset. The workflow demonstrates data preprocessing, training of multiple regression models, hyperparameter tuning, and continuous integration using GitHub Actions.

## Project Structure

```

HousingRegression/
├── .github/workflows/ci.yml      # GitHub Actions CI workflow
├── utils.py                      # Utility functions for data loading and evaluation
├── regression.py                 # Regression models and hyperparameter tuning logic
├── requirements.txt              # Python package dependencies
└── README.md                     # Project documentation

```

## Setup Instructions

### 1. Clone the Repository

```

git clone https://github.com/Pranav-Sri-Vasthav-Tenali-Gnana/HousingRegression.git
cd HousingRegression

```

### 2. Create and Activate Virtual Environment

```

python -m venv venv
venv\Scripts\activate        # For Windows

# or

source venv/bin/activate     # For macOS/Linux

```

### 3. Install Dependencies

```

pip install --upgrade pip
pip install -r requirements.txt

```

## Running the Project

### Regression Branch

The `reg` branch contains the implementation of three basic regression models:

- Ridge Regression
- Decision Tree Regressor
- Random Forest Regressor

To run:

```

git checkout reg
python regression.py

```

This will print the model performance using Mean Squared Error (MSE) and R² score.

### Hyperparameter Tuning Branch

The `hyper` branch includes the same models, extended with hyperparameter tuning using `GridSearchCV`.

To run:

```

git checkout hyper
python regression.py

```

This will output the best parameters along with the updated MSE and R² score for each model.

## Continuous Integration with GitHub Actions

A GitHub Actions CI workflow is configured to:

- Install dependencies from `requirements.txt`
- Run `regression.py` to ensure the workflow executes successfully

The workflow is located at `.github/workflows/ci.yml` and runs automatically on every push to the `main`, `reg`, and `hyper` branches.

## Notes

- The Boston Housing dataset is loaded manually from [StatLib](http://lib.stat.cmu.edu/datasets/boston) due to its removal from scikit-learn.
- The project maintains three branches as required:
  - `main`
  - `reg`
  - `hyper`
- All Git operations were performed through the command line interface.

## Author

**Name**: Pranav Sri Vasthav Tenali Gnana  
**Roll Number**: G24AI1114  
**Institute**: Indian Institute of Technology Jodhpur
```