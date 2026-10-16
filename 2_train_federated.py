import os
import pandas as pd
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
import joblib

# 0. تحديد مسار فولدر fl_project أوتوماتيكياً
folder_path = os.path.dirname(os.path.abspath(__file__))

# 1. تحميل داتا الاختبار المركزية (اللي السيرفر هيقيس بيها الدقة)
test_df = pd.read_csv(os.path.join(folder_path, "global_test.csv"))
X_test = test_df[['Age', 'Year', 'Nodes']].values
y_test = test_df['Survival'].values

# 2. تحميل داتا الـ 3 مستشفيات المعزولة
hospitals_data = []
for i in [1, 2, 3]:
    df_h = pd.read_csv(os.path.join(folder_path, f"hospital_{i}.csv"))
    X_h = df_h[['Age', 'Year', 'Nodes']].values
    y_h = df_h['Survival'].values
    hospitals_data.append((X_h, y_h))

print("\n" + "="*60)
print("📌 أولاً: دقة كل مستشفى لو اتدربت لوحدها (Local Models)")
print("="*60)
for i, (X_h, y_h) in enumerate(hospitals_data, 1):
    local_only = SGDClassifier(loss='log_loss', max_iter=100, random_state=42)
    local_only.fit(X_h, y_h)
    acc = accuracy_score(y_test, local_only.predict(X_test))
    print(f"🏥 مستشفى ({i}) منفردة: {acc*100:.2f}%")

print("\n" + "="*60)
print("🚀 ثانياً: بدء التدريب الفيدرالي (Federated Learning - FedAvg)")
print("="*60)

# إعدادات الشبكة الفيدرالية (10 جولات اتصال)
num_rounds = 10
global_weights = np.zeros((1, 3)) # 3 أوزان للأعمدة: Age, Year, Nodes
global_intercept = np.zeros((1,))
classes = np.array([0, 1])
total_samples = sum(len(x) for x, _ in hospitals_data)

for r in range(1, num_rounds + 1):
    local_weights_list = []
    local_intercepts_list = []
    sample_ratios = []

    # --- أ) التدريب المحلي داخل كل مستشفى ---
    for i, (X_h, y_h) in enumerate(hospitals_data, 1):
        client_model = SGDClassifier(
            loss='log_loss', learning_rate='constant', eta0=0.05, 
            max_iter=1, warm_start=True, random_state=42+r
        )
        
        # استلام أوزان السيرفر الحالية
        client_model.coef_ = global_weights.copy()
        client_model.intercept_ = global_intercept.copy()
        
        # تدريب على داتا المستشفى فقط
        client_model.partial_fit(X_h, y_h, classes=classes)
        
        # إرسال الأوزان فقط للسيرفر
        local_weights_list.append(client_model.coef_)
        local_intercepts_list.append(client_model.intercept_)
        sample_ratios.append(len(X_h) / total_samples)

    # --- ب) تجميع الأوزان في السيرفر (FedAvg Aggregation) ---
    global_weights = np.average(local_weights_list, axis=0, weights=sample_ratios)
    global_intercept = np.average(local_intercepts_list, axis=0, weights=sample_ratios)

    # --- ج) اختبار الموديل الفيدرالي العام ---
    global_model = SGDClassifier(loss='log_loss')
    global_model.coef_ = global_weights
    global_model.intercept_ = global_intercept
    global_model.classes_ = classes
    
    round_acc = accuracy_score(y_test, global_model.predict(X_test))
    print(f"🔄 جولة [{r:02d}/{num_rounds}] | دقة الموديل الفيدرالي: {round_acc*100:.2f}% | الأوزان الـ 3: {np.round(global_weights[0], 3)}")

# 3. حفظ الموديل الفيدرالي النهائي
joblib.dump(global_model, os.path.join(folder_path, "federated_model.pkl"))
print("\n✅ تم حفظ الموديل الفيدرالي النهائي في ملف: federated_model.pkl")