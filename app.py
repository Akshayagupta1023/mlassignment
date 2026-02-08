import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef, confusion_matrix, classification_report
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="Mobile Price Classification",
    page_icon="📱",
    layout="wide"
)

# Title
st.title("📱 Mobile Price Classification App")
st.markdown("""
This app predicts mobile phone price range (0-3) based on various specifications.
Upload your test data (CSV) or use the sample data to see predictions.
**Note**: Upload data WITHOUT the 'price_range' column for predictions.
""")

# Sidebar for navigation
st.sidebar.title("Navigation")
options = st.sidebar.radio("Select a page:", 
                          ["Home", "Upload & Predict", "Model Comparison", "About"])

# List of feature columns the models were trained on
FEATURE_COLUMNS = [
    'battery_power', 'blue', 'clock_speed', 'dual_sim', 'fc', 'four_g',
    'int_memory', 'm_dep', 'mobile_wt', 'n_cores', 'pc', 'px_height',
    'px_width', 'ram', 'sc_h', 'sc_w', 'talk_time', 'three_g',
    'touch_screen', 'wifi'
]

# Load saved models
@st.cache_resource
def load_models():
    try:
        models = {
            "Logistic Regression": joblib.load('saved_models/logistic_regression.pkl'),
            "Decision Tree": joblib.load('saved_models/decision_tree.pkl'),
            "K-Nearest Neighbors": joblib.load('saved_models/knn.pkl'),
            "Naive Bayes": joblib.load('saved_models/naive_bayes.pkl'),
            "Random Forest": joblib.load('saved_models/random_forest.pkl'),
            "XGBoost": joblib.load('saved_models/xgboost.pkl')
        }
        scaler = joblib.load('saved_models/scaler.pkl')
        return models, scaler
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None, None

# Home Page
if options == "Home":
    st.header("Welcome to Mobile Price Classifier")
    st.markdown("""
    ### Dataset Description
    This dataset contains information about 2000 mobile phones with the following features:
    
    **Features (20):**
    1. **battery_power** - Total energy a battery can store (mAh)
    2. **blue** - Has Bluetooth or not (0/1)
    3. **clock_speed** - Speed of microprocessor (GHz)
    4. **dual_sim** - Has dual sim support (0/1)
    5. **fc** - Front camera megapixels
    6. **four_g** - Has 4G or not (0/1)
    7. **int_memory** - Internal memory (GB)
    8. **m_dep** - Mobile depth (cm)
    9. **mobile_wt** - Weight of mobile phone
    10. **n_cores** - Number of cores of processor
    11. **pc** - Primary camera megapixels
    12. **px_height** - Pixel resolution height
    13. **px_width** - Pixel resolution width
    14. **ram** - Random Access Memory (MB)
    15. **sc_h** - Screen height (cm)
    16. **sc_w** - Screen width (cm)
    17. **talk_time** - Longest battery life on single charge (hours)
    18. **three_g** - Has 3G or not (0/1)
    19. **touch_screen** - Has touch screen or not (0/1)
    20. **wifi** - Has wifi or not (0/1)
    
    **Target:**
    - **price_range** - Price class (0: low cost, 1: medium cost, 2: high cost, 3: very high cost)
    
    ### Models Implemented
    1. Logistic Regression
    2. Decision Tree Classifier
    3. K-Nearest Neighbor Classifier
    4. Naive Bayes Classifier
    5. Random Forest (Ensemble)
    6. XGBoost (Ensemble)
    """)
    
    # Show sample data
    if st.checkbox("Show sample data"):
        try:
            sample_data = pd.read_csv('mobile_price.csv').head(10)
            st.dataframe(sample_data)
        except:
            st.info("Sample data not available locally. Use the uploaded data feature.")

# Upload & Predict Page
elif options == "Upload & Predict":
    st.header("📤 Upload Test Data and Predict")
    
    # Important note
    st.info("**Important**: Upload CSV file with only the 20 feature columns (without 'price_range' column).")
    
    # Option 1: Upload CSV
    uploaded_file = st.file_uploader("Upload your test CSV file (WITHOUT price_range column)", type=['csv'])
    
    # Option 2: Use sample test data
    use_sample = st.checkbox("Use sample test data (last 100 rows without price_range)")
    
    if uploaded_file is not None or use_sample:
        try:
            # Load data
            if uploaded_file is not None:
                test_df = pd.read_csv(uploaded_file)
            else:
                # Use last 100 rows as sample test data (without price_range)
                full_data = pd.read_csv('mobile_price.csv')
                test_df = full_data.drop('price_range', axis=1).tail(100)
            
            st.success("✅ Data loaded successfully!")
            st.write(f"**Shape:** {test_df.shape}")
            st.write("**First 5 rows:**")
            st.dataframe(test_df.head())
            
            # Check for price_range column (should NOT be present)
            if 'price_range' in test_df.columns:
                st.warning("⚠️ 'price_range' column found in uploaded data. This column will be removed for prediction.")
                test_df = test_df.drop('price_range', axis=1)
            
            # Check if all required columns are present
            missing_cols = [col for col in FEATURE_COLUMNS if col not in test_df.columns]
            extra_cols = [col for col in test_df.columns if col not in FEATURE_COLUMNS]
            
            if missing_cols:
                st.error(f"❌ Missing required columns: {missing_cols}")
                st.info("Please ensure your CSV has all 20 feature columns.")
            else:
                if extra_cols:
                    st.warning(f"⚠️ Extra columns found: {extra_cols}. Only the 20 required features will be used.")
                
                # Model selection
                st.subheader("Select Model for Prediction")
                model_names = ["Logistic Regression", "Decision Tree", "K-Nearest Neighbors", 
                              "Naive Bayes", "Random Forest", "XGBoost"]
                selected_model = st.selectbox("Choose a model:", model_names)
                
                if st.button("🚀 Predict", type="primary"):
                    # Load models
                    models, scaler = load_models()
                    
                    if models is None:
                        st.error("Failed to load models. Please check if model files exist.")
                    else:
                        model = models[selected_model]
                        
                        # Select only the required feature columns in correct order
                        X_test = test_df[FEATURE_COLUMNS]
                        
                        # Make predictions
                        with st.spinner(f'Making predictions using {selected_model}...'):
                            try:
                                # Scale if KNN
                                if selected_model == "K-Nearest Neighbors":
                                    X_test_scaled = scaler.transform(X_test)
                                    predictions = model.predict(X_test_scaled)
                                else:
                                    predictions = model.predict(X_test)
                                
                                # Add predictions to dataframe
                                results_df = test_df.copy()
                                results_df['Predicted_Price_Range'] = predictions
                                
                                # Map price range to labels
                                price_labels = {0: "Low Cost", 1: "Medium Cost", 2: "High Cost", 3: "Very High Cost"}
                                results_df['Predicted_Label'] = results_df['Predicted_Price_Range'].map(price_labels)
                                
                                # Display results
                                st.subheader("📊 Prediction Results")
                                st.dataframe(results_df[['Predicted_Price_Range', 'Predicted_Label']].head(20))
                                
                                # Show distribution
                                st.subheader("📈 Prediction Distribution")
                                fig, ax = plt.subplots(figsize=(10, 6))
                                results_df['Predicted_Label'].value_counts().sort_index().plot(
                                    kind='bar', ax=ax, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
                                )
                                ax.set_title('Distribution of Predicted Price Ranges')
                                ax.set_xlabel('Price Range')
                                ax.set_ylabel('Count')
                                plt.xticks(rotation=45)
                                st.pyplot(fig)
                                
                                # Show confusion matrix if we have true labels (for sample data)
                                if uploaded_file is None and use_sample:
                                    # Get true labels for sample data
                                    true_labels = pd.read_csv('mobile_price.csv')['price_range'].tail(100).values
                                    
                                    st.subheader("📋 Confusion Matrix (Sample Data)")
                                    cm = confusion_matrix(true_labels, predictions)
                                    fig2, ax2 = plt.subplots(figsize=(8, 6))
                                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax2)
                                    ax2.set_xlabel('Predicted')
                                    ax2.set_ylabel('Actual')
                                    ax2.set_title('Confusion Matrix')
                                    st.pyplot(fig2)
                                    
                                    # Classification report
                                    st.subheader("📝 Classification Report (Sample Data)")
                                    report = classification_report(true_labels, predictions, target_names=['Low', 'Medium', 'High', 'Very High'])
                                    st.text(report)
                                
                                # Download results
                                csv = results_df.to_csv(index=False)
                                st.download_button(
                                    label="📥 Download Predictions as CSV",
                                    data=csv,
                                    file_name="mobile_price_predictions.csv",
                                    mime="text/csv"
                                )
                                
                            except Exception as e:
                                st.error(f"Prediction error: {str(e)}")
                    
        except Exception as e:
            st.error(f"❌ Error loading data: {str(e)}")
            st.info("Please check your CSV file format and columns.")

# Model Comparison Page
elif options == "Model Comparison":
    st.header("📊 Model Performance Comparison")
    
    # Load pre-calculated results
    try:
        results_df = pd.read_csv('model_comparison_results.csv')
        
        # Display metrics table
        st.subheader("Evaluation Metrics Table")
        
        # Format the dataframe before displaying (FIXED - no .style)
        display_df = results_df.copy()
        display_df = display_df.rename(columns={
            'Model': 'ML Model Name',
            'Accuracy': 'Accuracy',
            'AUC': 'AUC',
            'Precision': 'Precision',
            'Recall': 'Recall',
            'F1': 'F1 Score',
            'MCC': 'MCC Score'
        })
        
        # Format numeric columns to 4 decimal places
        numeric_cols = ['Accuracy', 'AUC', 'Precision', 'Recall', 'F1 Score', 'MCC Score']
        for col in numeric_cols:
            if col in display_df.columns:
                display_df[col] = display_df[col].apply(lambda x: f"{x:.4f}")
        
        st.dataframe(display_df)
        
        # Create comparison chart
        st.subheader("Model Performance Visualization")
        
        # Let user select metric to visualize
        metric_to_plot = st.selectbox("Select metric to visualize:", 
                                     ['Accuracy', 'AUC', 'Precision', 'Recall', 'F1', 'MCC'])
        
        fig, ax = plt.subplots(figsize=(12, 6))
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFD166', '#118AB2']
        bars = ax.bar(results_df['Model'], results_df[metric_to_plot], color=colors)
        ax.set_title(f'{metric_to_plot} Comparison Across Models', fontsize=16, fontweight='bold')
        ax.set_xlabel('Model', fontsize=12)
        ax.set_ylabel(metric_to_plot, fontsize=12)
        ax.set_ylim([0, 1.05])
        plt.xticks(rotation=45, ha='right')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{height:.3f}', ha='center', va='bottom', fontweight='bold')
        
        st.pyplot(fig)
        
        # Show observations in a table format
        st.subheader("🔍 Model Performance Observations")
        
        observations_data = {
            'ML Model Name': [
                'Logistic Regression',
                'Decision Tree',
                'K-Nearest Neighbors',
                'Naive Bayes',
                'Random Forest (Ensemble)',
                'XGBoost (Ensemble)'
            ],
            'Observation about model performance': [
                'Achieved 67% accuracy, providing a reasonable baseline as a linear model. While it captures global patterns effectively, it struggles with complex non-linear relationships present in the data.',
                'Performed well with 83% accuracy, significantly better than Logistic Regression. However, the AUC score (0.8867) being lower than accuracy suggests some overfitting.',
                'Achieved only 50% accuracy (equivalent to random guessing for 4 classes), indicating poor performance. This suggests the distance metrics struggle with the 20-dimensional feature space.',
                'Performed surprisingly well with 81% accuracy despite the naive independence assumption. The high AUC (0.9506) indicates excellent ranking capability.',
                'Demonstrated strong performance with 88% accuracy, significantly outperforming the single Decision Tree. The ensemble approach effectively reduces overfitting.',
                'Achieved the best performance with 92.25% accuracy and near-perfect AUC (0.9937). The gradient boosting framework sequentially corrects errors, capturing complex patterns in the data.'
            ]
        }
        
        observations_df = pd.DataFrame(observations_data)
        st.dataframe(observations_df)
        
    except Exception as e:
        st.error(f"Error loading results: {str(e)}")
        st.info("Please run the model training notebook first to generate comparison results.")
        
        # Show sample data if results file doesn't exist
        if st.checkbox("Show sample metrics for testing"):
            sample_data = {
                'ML Model Name': ['Logistic Regression', 'Decision Tree', 'K-Nearest Neighbors', 
                                 'Naive Bayes', 'Random Forest', 'XGBoost'],
                'Accuracy': [0.6700, 0.8300, 0.5000, 0.8100, 0.8800, 0.9225],
                'AUC': [0.8968, 0.8867, 0.7697, 0.9506, 0.9769, 0.9937],
                'Precision': [0.6809, 0.8319, 0.5211, 0.8113, 0.8796, 0.9226],
                'Recall': [0.6700, 0.8300, 0.5000, 0.8100, 0.8800, 0.9225],
                'F1': [0.6746, 0.8302, 0.5054, 0.8105, 0.8797, 0.9225],
                'MCC': [0.5605, 0.7738, 0.3350, 0.7468, 0.8400, 0.8967]
            }
            sample_df = pd.DataFrame(sample_data)
            st.dataframe(sample_df)

# About Page
elif options == "About":
    st.header("ℹ️ About This Project")
    st.markdown("""
    ### Machine Learning Assignment 2
    
    **Objective**: Implement and compare 6 classification models for mobile price prediction.
    
    **Models Implemented**:
    1. Logistic Regression
    2. Decision Tree Classifier
    3. K-Nearest Neighbor Classifier
    4. Naive Bayes Classifier
    5. Random Forest (Ensemble)
    6. XGBoost (Ensemble)
    
    **Evaluation Metrics**:
    - Accuracy
    - AUC Score
    - Precision
    - Recall
    - F1 Score
    - Matthews Correlation Coefficient (MCC)
    
    **Dataset**: Mobile Price Classification Dataset (2000 samples, 20 features)
    
    **Deployment**: Streamlit Web Application
    
    ### How to Use
    1. Go to **Upload & Predict** page to test the models
    2. Upload a CSV file with mobile features (**without price_range column**)
    3. Select a model from dropdown
    4. Click Predict to see results
    5. Check **Model Comparison** page to see performance metrics
    
    ### Technical Details
    - Built with Python, Scikit-learn, XGBoost
    - Web interface: Streamlit
    - Deployment: Streamlit Community Cloud
    - Environment: BITS Pilani Virtual Lab
    """)
    
    st.info("**Note**: For assignment submission, this app is deployed on Streamlit Community Cloud with all required features.")

# Footer
st.sidebar.markdown("---")
st.sidebar.info(
    """
    **Assignment 2 - Machine Learning**  
    M.Tech (AIML/DSE)  
    BITS Pilani WILP  
    **Submission**: 15-Feb-2026
    """
)

# Add required features checklist in sidebar
st.sidebar.markdown("---")
st.sidebar.subheader("✅ Assignment Features")
st.sidebar.markdown("""
- [x] Dataset upload option (CSV)
- [x] Model selection dropdown
- [x] Display of evaluation metrics  
- [x] Confusion matrix / Classification report
- [x] 6 ML models implemented
- [x] All metrics calculated
- [x] Deployed on Streamlit Cloud
""")