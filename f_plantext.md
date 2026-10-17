يا هندسة دي ملفات الـ AI Model API جاهزة (حط الـ 3 ملفات دول في نفس الفولدر عندك):
1. 4_api.py
2. federated_model.pkl
3. haberman.csv

🔹 خطوات التشغيل عندك في الـ Terminal:
1️⃣ سطب المكتبات بالأمر ده:
pip install fastapi uvicorn pandas scikit-learn joblib

2️⃣ شغل سيرفر الـ API بالأمر ده (وأنت واقف جوه الفولدر):
python -m uvicorn 4_api:app --reload

-----------------------------------------
🔹 تفاصيل الربط (API Documentation):

📌 Endpoint URL:
POST http://127.0.0.1:8000/predict

📌 Swagger UI (عشان تجرب الـ API في المتصفح):
http://127.0.0.1:8000/docs

📌 Request Body (JSON) اللي هتبعته للـ API:
{
  "Age": 45,
  "Year": 64,
  "Nodes": 2
}
(ملحوظة: Age هو عمر المريض، Year هي سنة العملية من رقمين زي 64، و Nodes هي عدد العقد المصابة).

📌 Response (JSON) اللي هيرجعلك من الموديل:
{
  "prediction": 1,
  "status": "Survived (>= 5 years)",
  "survival_probability": "78.5%"
}
(ملحوظة: prediction بيرجع 1 لو المريض هيتعافى، و 0 لو الحالة حرجة).