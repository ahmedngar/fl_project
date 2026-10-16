import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Federated Learning Medical App", layout="wide")

st.title("🏥 نظام التعلم الفيدرالي لتوقع نجاة مرضى الجراحة (Federated Learning)")
st.markdown("تم تدريب هذا النموذج بالتعاون بين **3 مستشفيات منفصلة** باستخدام خوارزمية **FedAvg** دون مشاركة بيانات المرضى الخام.")

folder_path = os.path.dirname(os.path.abspath(__file__))

# 1. تحميل الموديل الفيدرالي والبيانات
model = joblib.load(os.path.join(folder_path, "federated_model.pkl"))
raw_df = pd.read_csv(os.path.join(folder_path, "haberman.csv"))
test_df = pd.read_csv(os.path.join(folder_path, "global_test.csv"))

# تجهيز الـ Scaler على البيانات الأصلية لتحويل مدخلات المستخدم الجديدة
scaler = StandardScaler()
scaler.fit(raw_df[['Age', 'Year', 'Nodes']])

# حساب دقة كل مستشفى لوحده مقارنة بالموديل الفيدرالي للعرض
X_test = test_df[['Age', 'Year', 'Nodes']].values
y_test = test_df['Survival'].values

accuracies = {}
for i in [1, 2, 3]:
    df_h = pd.read_csv(os.path.join(folder_path, f"hospital_{i}.csv"))
    local_m = SGDClassifier(loss='log_loss', max_iter=100, random_state=42)
    local_m.fit(df_h[['Age', 'Year', 'Nodes']].values, df_h['Survival'].values)
    accuracies[f"Hospital {i} (Local)"] = accuracy_score(y_test, local_m.predict(X_test)) * 100

accuracies["Federated Global Model"] = accuracy_score(y_test, model.predict(X_test)) * 100

# --- تقسيم الشاشة لعمودين ---
col1, col2 = st.columns([1.2, 1])

with col1:
    st.subheader("📊 مقارنة الدقة: التدريب المنفرد vs الفيدرالي")
    acc_df = pd.DataFrame(list(accuracies.items()), columns=["Model", "Accuracy (%)"])
    st.bar_chart(acc_df.set_index("Model"))
    
    st.subheader("⚖️ أوزان النموذج الفيدرالي النهائية (Feature Importance)")
    weights_df = pd.DataFrame({
        "Feature": ["Age (العمر)", "Year (سنة العملية)", "Nodes (العقد المصابة)"],
        "Weight": model.coef_[0]
    })
    st.dataframe(weights_df, use_container_width=True)

with col2:
    st.subheader("🩺 فحص حالة مريض جديد (Inference)")
    age = st.slider("عمر المريض (Age)", min_value=20, max_value=90, value=45)
    year = st.slider("سنة إجراء العملية (19XX)", min_value=58, max_value=70, value=64)
    nodes = st.number_input("عدد العقد الليمفاوية المصابة (Nodes)", min_value=0, max_value=50, value=1)

    if st.button("🔍 توقع الحالة الآن (Predict)"):
        # عمل Scaling لبيانات المريض الجديد
        input_data = pd.DataFrame([[age, year, nodes]], columns=['Age', 'Year', 'Nodes'])
        input_scaled = scaler.transform(input_data)
        
        pred = model.predict(input_scaled)[0]
        prob = model.predict_proba(input_scaled)[0][1] * 100

        if pred == 1:
            st.success(f"✅ توقع النموذج: تعافي المريض وبقاؤه لأكثر من 5 سنوات (بنسبة ثقة: {prob:.1f}%)")
        else:
            st.error(f"⚠️ توقع النموذج: حالة حرجة عالية الخطورة (فرصة التعافي: {prob:.1f}% فقط)")