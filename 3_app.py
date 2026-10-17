import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc

st.set_page_config(page_title="Federated Learning Healthcare", layout="wide")

folder_path = os.path.dirname(os.path.abspath(__file__))

# تحميل الداتا والموديلات وجدول المقاييس
df_orig = pd.read_csv(os.path.join(folder_path, "haberman.csv"))
test_df = pd.read_csv(os.path.join(folder_path, "global_test.csv"))
model = joblib.load(os.path.join(folder_path, "federated_model.pkl"))

scaler = StandardScaler()
scaler.fit(df_orig[['Age', 'Year', 'Nodes']])

st.title("🏥 نظام التنبؤ الطبي باستخدام التعلم الفيدرالي (Federated Learning)")
st.markdown("مقارنة شاملة بين **التدريب المحلي المنفرد (Local)**، **التدريب المركزي (Centralized)**، و**التدريب الفيدرالي (FedAvg)**.")

# --- القسم الأول: جدول ورسومات المقارنة الشاملة ---
metrics_path = os.path.join(folder_path, "metrics_comparison.csv")
if os.path.exists(metrics_path):
    metrics_df = pd.read_csv(metrics_path)
    
    st.subheader("📊 1. جدول تقييم الأداء الشامل (Evaluation Metrics)")
    st.dataframe(metrics_df, use_container_width=True)

    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 مقارنة المقاييس (Local vs. Centralized vs. Federated)")
        metric_choice = st.selectbox("اختر المقياس لعرضه في الرسم البياني:", ["Accuracy", "F1-Score", "Precision", "Recall", "ROC-AUC"])
        st.bar_chart(metrics_df.set_index("Model")[metric_choice])

    with col2:
        st.subheader("🎯 منحنى ROC-AUC (Federated vs. Centralized)")
        X_test = test_df[['Age', 'Year', 'Nodes']].values
        y_test = test_df['Survival'].values
        
        fig, ax = plt.subplots(figsize=(6, 4))
        
        # منحنى الموديل الفيدرالي
        y_prob_fed = model.predict_proba(X_test)[:, 1]
        fpr_fed, tpr_fed, _ = roc_curve(y_test, y_prob_fed)
        ax.plot(fpr_fed, tpr_fed, label=f"Federated Model (AUC = {auc(fpr_fed, tpr_fed):.2f})", linewidth=2)
        
        # منحنى الموديل المركزي إن وجد
        central_path = os.path.join(folder_path, "centralized_model.pkl")
        if os.path.exists(central_path):
            central_model = joblib.load(central_path)
            y_prob_cent = central_model.predict_proba(X_test)[:, 1]
            fpr_c, tpr_c, _ = roc_curve(y_test, y_prob_cent)
            ax.plot(fpr_c, tpr_c, linestyle="--", label=f"Centralized Model (AUC = {auc(fpr_c, tpr_c):.2f})", linewidth=2)
            
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        ax.set_xlabel("False Positive Rate (FPR)")
        ax.set_ylabel("True Positive Rate (TPR)")
        ax.set_title("ROC Curve Comparison")
        ax.legend(loc="lower right")
        st.pyplot(fig)

st.divider()

# --- القسم الثاني: تجربة التوقع لمريض جديد ---
st.subheader("🩺 2. فحص حالة مريض جديد (Real-Time Prediction)")
c1, c2, c3 = st.columns(3)
with c1:
    age = st.number_input("عمر المريض (Age)", min_value=20, max_value=100, value=45)
with c2:
    year = st.number_input("سنة العملية (Year: 58-70)", min_value=58, max_value=70, value=64)
with c3:
    nodes = st.number_input("عدد العقد المصابة (Nodes)", min_value=0, max_value=60, value=2)

if st.button("🔍 توقع الحالة الآن (Predict)"):
    input_data = pd.DataFrame([[age, year, nodes]], columns=['Age', 'Year', 'Nodes'])
    input_scaled = scaler.transform(input_data)
    
    pred = model.predict(input_scaled)[0]
    prob = model.predict_proba(input_scaled)[0][1] * 100
    
    if pred == 1:
        st.success(f"✅ توقع النموذج: تعافي المريض وبقاؤه لأكثر من 5 سنوات (بنسبة ثقة: {prob:.1f}%)")
    else:
        st.error(f"⚠️ توقع النموذج: حالة حرجة عالية الخطورة (فرصة التعافي: {prob:.1f}% فقط)")