# Mobile Price Classification - Machine Learning Assignment 2

## 📋 Problem Statement
Predict the price range of mobile phones (0: low cost, 1: medium cost, 2: high cost, 3: very high cost) based on their technical specifications and features.

## 📊 Dataset Description
**Source**: Kaggle - Mobile Price Classification Dataset  
**Samples**: 2000 mobile phones  
**Features**: 20 technical specifications  
**Target**: price_range (4 classes: 0, 1, 2, 3)

### Features: [Same as before - keep the feature list]

## 🤖 Models Implemented
1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbor Classifier
4. Naive Bayes Classifier (Gaussian)
5. Random Forest (Ensemble)
6. XGBoost (Ensemble)

## 📈 Evaluation Metrics Comparison

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---------------|----------|-----|-----------|--------|----|-----|
| Logistic Regression | {FILL_THIS} | {FILL_THIS} | {FILL_THIS} | {FILL_THIS} | {FILL_THIS} | {FILL_THIS} |
| Decision Tree | {FILL_THIS} | {FILL_THIS} | {FILL_THIS} | {FILL_THIS} | {FILL_THIS} | {FILL_THIS} |
| K-Nearest Neighbors | 0.5000 | 0.7697 | 0.5211 | 0.5000 | 0.5054 | 0.3350 |
| Naive Bayes | 0.8100 | 0.9506 | 0.8113 | 0.8100 | 0.8105 | 0.7468 |
| Random Forest (Ensemble) | 0.8800 | 0.9769 | 0.8796 | 0.8800 | 0.8797 | 0.8400 |
| XGBoost (Ensemble) | 0.9225 | 0.9937 | 0.9226 | 0.9225 | 0.9225 | 0.8967 |

## 🔍 Observations about Model Performance

| ML Model Name | Observation about model performance |
|---------------|--------------------------------------|
| Logistic Regression | [Based on your results: e.g., "Achieved moderate accuracy around XX%, suitable as a baseline linear model."] |
| Decision Tree | [Based on your results: e.g., "Showed accuracy of XX%, likely overfitting due to perfect training scores."] |
| K-Nearest Neighbors | Achieved only 50% accuracy, indicating poor performance possibly due to the curse of dimensionality with 20 features. The model struggled to find meaningful neighbors in the feature space. |
| Naive Bayes | Performed surprisingly well with 81% accuracy despite the independence assumption. The Gaussian distribution fit the continuous features appropriately. |
| Random Forest (Ensemble) | Demonstrated strong performance with 88% accuracy, significantly better than a single Decision Tree, showing the power of ensemble methods in reducing variance. |
| XGBoost (Ensemble) | Achieved the best performance with 92.25% accuracy and 0.9937 AUC, demonstrating state-of-the-art capability for tabular data classification through gradient boosting. |

## 🚀 Streamlit Application Features

The interactive web application includes all required features:

1. **Dataset upload option** - Users can upload CSV files containing mobile features for prediction (test data only as per assignment requirements)
2. **Model selection dropdown** - Interactive dropdown to select from all 6 implemented models
3. **Display of evaluation metrics** - Complete comparison table showing all evaluation metrics for each model
4. **Visualizations** - Interactive charts showing model performance comparison and prediction distributions
5. **Results download** - Option to download predictions as CSV file
6. **Confusion Matrix/Classification Report** - Visual representation of model performance on test data

### Application Pages:
- **Home**: Dataset description and sample data visualization
- **Upload & Predict**: Interactive prediction interface with model selection
- **Model Comparison**: Detailed performance metrics and visual comparisons
- **About**: Project information and technical details

## 🛠️ Installation and Usage

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Local Installation
```bash
# Clone the repository
git clone https://github.com/your-username/mobile-price-classification.git
cd mobile-price-classification

# Install dependencies
pip install -r requirements.txt