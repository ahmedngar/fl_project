import os
import joblib
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sklearn.preprocessing import StandardScaler

app = FastAPI(title="Federated Model API")

# السماح للباك إند بالاتصال بدون مشاكل CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# قراءة ملف الموديل والداتا من نفس فولدر fl_project
folder_path = os.path.dirname(os.path.abspath(__file__))
model = joblib.load(os.path.join(folder_path, "federated_model.pkl"))
df_orig = pd.read_csv(os.path.join(folder_path, "haberman.csv"))

# تجهيز الـ Scaler من الداتا الأصلية
scaler = StandardScaler()
scaler.fit(df_orig[['Age', 'Year', 'Nodes']])

# شكل البيانات اللي الباك إند هيبعتها
class PatientData(BaseModel):
    Age: int
    Year: int
    Nodes: int

@app.get("/")
def home():
    return {"message": "Federated Model API is working!"}

@app.post("/predict")
def predict(data: PatientData):
    # تحويل الأرقام وعمل Scaling
    input_df = pd.DataFrame([[data.Age, data.Year, data.Nodes]], columns=['Age', 'Year', 'Nodes'])
    input_scaled = scaler.transform(input_df)
    
    # التوقع من الموديل المحفوظ
    pred = int(model.predict(input_scaled)[0])
    prob = round(float(model.predict_proba(input_scaled)[0][1]) * 100, 2)
    
    return {
        "prediction": pred,
        "status": "Survived (>= 5 years)" if pred == 1 else "High Risk (< 5 years)",
        "survival_probability": f"{prob}%"
    }