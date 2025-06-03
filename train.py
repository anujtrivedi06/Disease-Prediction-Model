import pandas as pd
import numpy as np
from sklearn import datasets
import seaborn as sns
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.feature_selection import SelectKBest, chi2
from imblearn.over_sampling import RandomOverSampler
from sklearn import metrics
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression

# Loading the dataset
df = pd.read_csv("parkinson_disease.csv")

# print(df.head())

# DAta preprocessing 

# Grouping by ID
df = df.groupby('id').mean().reset_index()
df.drop('id', axis=1, inplace=True)

# Checking for NULL values
print(df.isnull().sum().sum())
#There are no NaN values

print(f"Original number of features are: {df.shape[1]}")

# Removing highly correltae columns
columns = list(df.columns)
for col in columns:
    if col == "class":
        continue

    filtered_columns = [col]
    for col1 in df.columns:
        if((col == col1) | (col == "class")):
            continue
        val = df[col].corr(df[col1])
        if val>0.7:
            columns.remove(col1)
            continue
        else:
            filtered_columns.append(col1)
    df = df[filtered_columns]

print(f"Reduced number of features are: {df.shape[1]}")

# Features reduced from 754 to 287

#Applyig the Chi Square Feature Selection to use only required featured
X = df.drop('class', axis=1)
X_norm = MinMaxScaler().fit_transform(X)
selector = SelectKBest(chi2, k=30)
selector.fit(X_norm, df['class'])
filtered_columns = selector.get_support()
filtered_data = X.loc[:, filtered_columns]
filtered_data['class'] = df['class']
df = filtered_data
print(df.shape)


# Creating the Pie Chart to visualize the output class
x = df['class'].value_counts()
plt.pie(x.values,
        labels = x.index,
        autopct='%1.1f%%')
plt.show()


features = df.drop('class', axis=1)
target = df['class']

X_train, X_val,Y_train, Y_val = train_test_split(features, target, test_size=0.2, random_state=10)

# Sampling the minority class("0") to same no. as the majority class("1")
ros = RandomOverSampler(sampling_strategy=1.0, random_state=0)
X_resampled, y_resampled = ros.fit_resample(X_train, Y_train)
print(f"The X shape : {X_resampled.shape}")
print(f"Y value counts : {y_resampled.value_counts()}")




# Making predictions 
from sklearn.metrics import roc_auc_score as ras

models = [LogisticRegression(class_weight='balanced'), XGBClassifier(), SVC(kernel='rbf', probability=True)] 
for model in models:
    model.fit(X_resampled, y_resampled)
    print(f'{model} : ')

    train_preds = model.predict(X_resampled)
    print('Training Accuracy : ', ras(y_resampled, train_preds))

    val_preds = model.predict(X_val)
    print('Validation Accuracy : ', ras(Y_val, val_preds))
    print()

# Printing the confusion matrix for every model
from sklearn.metrics import ConfusionMatrixDisplay

for model in models:
    ConfusionMatrixDisplay.from_estimator(model, X_val, Y_val)
    plt.title(f"{model}")
    plt.show()

from sklearn.metrics import classification_report
print(classification_report(Y_val, models[0].predict(X_val)))
