# Parkinson's Disease Detection using Machine Learning

This project focuses on the classification of Parkinson's Disease using a dataset of biomedical voice measurements. The goal is to build models that accurately differentiate between healthy and affected individuals.

## 📁 Dataset

The dataset used is `parkinson_disease.csv`, which contains biomedical voice measurements from patients. Each row represents a set of features extracted.

### Key Information:
- **Target Column**: `class` (0 = Healthy, 1 = Parkinson's Disease)
- **Features**: Multiple numerical columns representing voice signal features.

## 🧹 Data Preprocessing

1. **ID Grouping**: Data grouped by `id` and mean values taken.
2. **Null Values**: Checked and confirmed no nulls.
3. **Feature Reduction via Correlation**: Highly correlated features (correlation > 0.7) are removed.
4. **Chi-Square Feature Selection**: Top 30 features selected using the Chi-Square test.
5. **Data Normalization**: MinMaxScaler used to scale features.
6. **Class Balancing**: Applied `RandomOverSampler` to balance the dataset.

## 📊 Visualizations

- A **pie chart** displays the class distribution before resampling.
- **Confusion matrices** for each model visualize classification performance.

## 🤖 Models Used

Three machine learning models were trained and evaluated:

1. **Logistic Regression**
2. **XGBoost Classifier**
3. **Support Vector Machine (RBF Kernel)**

All models are evaluated using:
- **ROC-AUC Score** for training and validation sets
- **Confusion Matrix**
- **Classification Report** (for Logistic Regression)

## 🧪 Evaluation Metrics

- **ROC-AUC Score**
- **Accuracy**
- **Precision, Recall, F1-Score**
- **Confusion Matrix**

## 📝 Requirements

Install required packages before running the code:

```bash
pip install pandas numpy scikit-learn seaborn matplotlib imbalanced-learn xgboost
