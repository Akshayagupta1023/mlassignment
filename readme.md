# Mobile Price Classification - Machine Learning Assignment 2

## 📋 Problem Statement
Predict the price range of mobile phones (0: low cost, 1: medium cost, 2: high cost, 3: very high cost) based on their technical specifications and features. This is a multi-class classification problem with four target classes.

## 📊 Dataset Description
**Source**: Kaggle - Mobile Price Classification Dataset  
**Samples**: 2000 mobile phones  
**Features**: 20 technical specifications  
**Target**: price_range (4 classes: 0, 1, 2, 3)  
**Dataset Characteristics**: Balanced classes with 500 samples per price range

### Features:
1. **battery_power** - Total energy a battery can store in mAh
2. **blue** - Bluetooth availability (0: No, 1: Yes)
3. **clock_speed** - Microprocessor speed in GHz
4. **dual_sim** - Dual SIM support (0: No, 1: Yes)
5. **fc** - Front camera megapixels
6. **four_g** - 4G support (0: No, 1: Yes)
7. **int_memory** - Internal memory in GB
8. **m_dep** - Mobile depth in cm
9. **mobile_wt** - Weight of mobile phone
10. **n_cores** - Number of processor cores
11. **pc** - Primary camera megapixels
12. **px_height** - Pixel resolution height
13. **px_width** - Pixel resolution width
14. **ram** - Random Access Memory in MB
15. **sc_h** - Screen height in cm
16. **sc_w** - Screen width in cm
17. **talk_time** - Longest battery life on single charge (hours)
18. **three_g** - 3G support (0: No, 1: Yes)
19. **touch_screen** - Touch screen availability (0: No, 1: Yes)
20. **wifi** - WiFi availability (0: No, 1: Yes)

**Target Variable**:
- **price_range**: 
  - 0: Low Cost
  - 1: Medium Cost
  - 2: High Cost
  - 3: Very High Cost

## 🤖 Models Implemented
Six classification models were implemented and evaluated on the same dataset:

1. **Logistic Regression** - Linear classification model
2. **Decision Tree Classifier** - Non-linear tree-based model
3. **K-Nearest Neighbor Classifier** - Instance-based learning
4. **Naive Bayes Classifier (Gaussian)** - Probabilistic classifier
5. **Random Forest (Ensemble)** - Bagging ensemble of decision trees
6. **XGBoost (Ensemble)** - Gradient boosting ensemble

## 📈 Evaluation Metrics Comparison

All models were evaluated using the following metrics on a 20% test set (400 samples):

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---------------|----------|-----|-----------|--------|----|-----|
| Logistic Regression | 0.6700 | 0.8968 | 0.6809 | 0.6700 | 0.6746 | 0.5605 |
| Decision Tree | 0.8300 | 0.8867 | 0.8319 | 0.8300 | 0.8302 | 0.7738 |
| K-Nearest Neighbors | 0.5000 | 0.7697 | 0.5211 | 0.5000 | 0.5054 | 0.3350 |
| Naive Bayes | 0.8100 | 0.9506 | 0.8113 | 0.8100 | 0.8105 | 0.7468 |
| Random Forest (Ensemble) | 0.8800 | 0.9769 | 0.8796 | 0.8800 | 0.8797 | 0.8400 |
| XGBoost (Ensemble) | 0.9225 | 0.9937 | 0.9226 | 0.9225 | 0.9225 | 0.8967 |

## 🔍 Observations about Model Performance

| ML Model Name | Observation about model performance |
|---------------|--------------------------------------|
| Logistic Regression | Achieved 67% accuracy, providing a reasonable baseline as a linear model. While it captures global patterns effectively, it struggles with complex non-linear relationships present in the data. The moderate MCC score (0.5605) indicates balanced performance across all classes. |
| Decision Tree | Performed well with 83% accuracy, significantly better than Logistic Regression. However, the AUC score (0.8867) being lower than accuracy suggests some overfitting. The tree structure effectively captures non-linear relationships but may benefit from pruning for better generalization. |
| K-Nearest Neighbors | Achieved only 50% accuracy (equivalent to random guessing for 4 classes), indicating poor performance. This suggests the distance metrics struggle with the 20-dimensional feature space, possibly due to the curse of dimensionality or need for better feature scaling/tuning of the k parameter. |
| Naive Bayes | Performed surprisingly well with 81% accuracy despite the naive independence assumption. The high AUC (0.9506) indicates excellent ranking capability. The Gaussian distribution fits the continuous features appropriately, and the model's simplicity provides fast training and prediction times. |
| Random Forest (Ensemble) | Demonstrated strong performance with 88% accuracy, significantly outperforming the single Decision Tree (83%). The ensemble approach effectively reduces overfitting through averaging, with excellent AUC (0.9769) indicating strong discriminative power across all classes. |
| XGBoost (Ensemble) | Achieved the best performance with 92.25% accuracy and near-perfect AUC (0.9937). The gradient boosting framework sequentially corrects errors, capturing complex patterns in the data. With the highest MCC (0.8967), it shows excellent balanced performance across all price categories. |

## 🚀 Streamlit Application Features

The interactive web application includes all required features:

1. **Dataset upload option** - Upload CSV files for prediction (test data only)
2. **Model selection dropdown** - Choose from all 6 implemented models
3. **Display of evaluation metrics** - Complete comparison table as shown above
4. **Visualizations** - Performance charts and prediction distributions
5. **Results download** - Export predictions as CSV
6. **Confusion Matrix/Classification Report** - Visual model evaluation

### Application Pages:
- **Home**: Dataset description and sample data visualization
- **Upload & Predict**: Interactive prediction interface with model selection
- **Model Comparison**: Performance metrics and visual comparisons
- **About**: Project information and technical details

## 🛠️ Installation and Usage

### Local Installation
```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
cd YOUR_REPO_NAME
pip install -r requirements.txt