import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, confusion_matrix
)

# Load the old dataset
data = pd.read_csv("/data/raw/bot_detection_data.csv")
X_old = data.drop(columns=["Bot Label"])
y_old = data["Bot Label"] 

# Load trained models
lgbm_model = joblib.load("models/lightgbm/lgbm_model.pkl")

# Make predictions
y_pred_lgbm = lgbm_model.predict(X_old)
y_pred_proba_lgbm = lgbm_model.predict_proba(X_old)[:, 1]  # Probability scores

# Evaluation metrics
metrics = {
    "Accuracy": accuracy_score(y_old, y_pred_lgbm),
    "Precision": precision_score(y_old, y_pred_lgbm),
    "Recall": recall_score(y_old, y_pred_lgbm),
    "F1-score": f1_score(y_old, y_pred_lgbm),
    "ROC-AUC": roc_auc_score(y_old, y_pred_proba_lgbm)
}

# Print evaluation results
print("\n🔍 Model Evaluation Results (Old Dataset):")
for metric, value in metrics.items():
    print(f"{metric}: {value:.4f}")

# Confusion Matrix Visualization
conf_matrix = confusion_matrix(y_old, y_pred_lgbm)
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["Human", "Bot"], yticklabels=["Human", "Bot"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Old Model")
plt.show()

# Bar Plot of Metrics
plt.figure(figsize=(8, 5))
sns.barplot(x=list(metrics.keys()), y=list(metrics.values()), palette="viridis")
plt.title("Model Performance on Old Dataset")
plt.ylim(0, 1)
plt.show()
