import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# 1. Load Data
df = pd.read_csv(r"C:\Users\jakku\Downloads\creditcard.csv.zip")
print(df.shape)
print(df['Class'].value_counts())

# 2. Handle Imbalanced Data - SMOTE or Undersampling
# We use undersampling for speed
fraud = df[df['Class']==1]
normal = df[df['Class']==0].sample(n=len(fraud)*5, random_state=42)
df_balanced = pd.concat([fraud, normal]).sample(frac=1, random_state=42)

X = df_balanced.drop('Class', axis=1)
y = df_balanced['Class']

# 3. Scale Amount and Time
scaler = StandardScaler()
X[['Time','Amount']] = scaler.fit_transform(X[['Time','Amount']])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. MODEL 1: Supervised - Logistic Regression
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
pred = model.predict(X_test)
print("Logistic Regression:")
print(classification_report(y_test, pred))
print("ROC-AUC:", roc_auc_score(y_test, pred))

# 5. MODEL 2: Unsupervised Anomaly Detection - Isolation Forest
# This works even WITHOUT labels - best for interview
iso = IsolationForest(contamination=0.01, random_state=42)
iso.fit(X_train)
anomaly_pred = iso.predict(X_test)
# Isolation Forest gives -1 for anomaly, 1 for normal. Convert to 0/1
anomaly_pred = [1 if x==-1 else 0 for x in anomaly_pred]
print("Isolation Forest - Anomaly Detection:")
print(classification_report(y_test, anomaly_pred))