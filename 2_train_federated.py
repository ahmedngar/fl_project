import os
import pandas as pd
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib

folder_path = os.path.dirname(os.path.abspath(__file__))

# 1. تحميل داتا الاختبار المركزية
test_df = pd.read_csv(os.path.join(folder_path, "global_test.csv"))
X_test = test_df[['Age', 'Year', 'Nodes']].values
y_test = test_df['Survival'].values

# دالة لحساب كل المقاييس اللي الدكتور طلبها
def evaluate_all_metrics(model, X, y, model_name):
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]
    return {
        "Model": model_name,
        "Accuracy": round(accuracy_score(y, y_pred) * 100, 2),
        "Precision": round(precision_score(y, y_pred, zero_division=0) * 100, 2),
        "Recall": round(recall_score(y, y_pred, zero_division=0) * 100, 2),
        "F1-Score": round(f1_score(y, y_pred, zero_division=0) * 100, 2),
        "ROC-AUC": round(roc_auc_score(y, y_prob) * 100, 2)
    }

results = []

# 2. تحميل داتا الـ 3 مستشفيات وتدريب كل مستشفى منفردة (Local Training)
hospitals_data = []
X_all_list, y_all_list = [], []

for i in [1, 2, 3]:
    df_h = pd.read_csv(os.path.join(folder_path, f"hospital_{i}.csv"))
    X_h = df_h[['Age', 'Year', 'Nodes']].values
    y_h = df_h['Survival'].values
    hospitals_data.append((X_h, y_h))
    X_all_list.append(X_h)
    y_all_list.append(y_h)

    # تدريب منفرد لكل مستشفى
    local_model = SGDClassifier(loss='log_loss', max_iter=100, random_state=42)
    local_model.fit(X_h, y_h)
    results.append(evaluate_all_metrics(local_model, X_test, y_test, f"Hospital {i} (Local)"))

# 3. التدريب المركزي (Centralized Training - تجميع كل الداتا في مكان واحد)
X_centralized = np.vstack(X_all_list)
y_centralized = np.concatenate(y_all_list)

central_model = SGDClassifier(loss='log_loss', max_iter=100, random_state=42)
central_model.fit(X_centralized, y_centralized)
results.append(evaluate_all_metrics(central_model, X_test, y_test, "Centralized Model (All Data)"))
joblib.dump(central_model, os.path.join(folder_path, "centralized_model.pkl"))

# 4. التدريب الفيدرالي (Federated Learning - FedAvg)
num_rounds = 10
global_weights = np.zeros((1, 3))
global_intercept = np.zeros((1,))
classes = np.array([0, 1])
total_samples = sum(len(x) for x, _ in hospitals_data)

for r in range(1, num_rounds + 1):
    local_weights_list, local_intercepts_list, sample_ratios = [], [], []

    for i, (X_h, y_h) in enumerate(hospitals_data, 1):
        client_model = SGDClassifier(
            loss='log_loss', learning_rate='constant', eta0=0.05, 
            max_iter=1, warm_start=True, random_state=42+r
        )
        client_model.coef_ = global_weights.copy()
        client_model.intercept_ = global_intercept.copy()
        client_model.partial_fit(X_h, y_h, classes=classes)
        
        local_weights_list.append(client_model.coef_)
        local_intercepts_list.append(client_model.intercept_)
        sample_ratios.append(len(X_h) / total_samples)

    global_weights = np.average(local_weights_list, axis=0, weights=sample_ratios)
    global_intercept = np.average(local_intercepts_list, axis=0, weights=sample_ratios)

global_model = SGDClassifier(loss='log_loss')
global_model.coef_ = global_weights
global_model.intercept_ = global_intercept
global_model.classes_ = classes

results.append(evaluate_all_metrics(global_model, X_test, y_test, "Federated Global Model (FedAvg)"))

# 5. حفظ الموديل الفيدرالي وجدول المقارنة الشامل
joblib.dump(global_model, os.path.join(folder_path, "federated_model.pkl"))
metrics_df = pd.DataFrame(results)
metrics_df.to_csv(os.path.join(folder_path, "metrics_comparison.csv"), index=False)

print("\n📊 جدول المقارنة الشامل (Local vs. Centralized vs. Federated):")
print("="*80)
print(metrics_df.to_string(index=False))
print("="*80)
print("✅ تم حفظ الموديلات وجدول المقاييس في metrics_comparison.csv بنجاح!")