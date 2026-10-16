import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 0. تحديد مسار الفولدر أوتوماتيكياً
folder_path = os.path.dirname(os.path.abspath(__file__))

# 1. قراءة الملف
df = pd.read_csv(os.path.join(folder_path, "haberman.csv"))

# 2. تعديل عمود الهدف ليكون 1 و 0
df['Survival'] = df['Survival'].map({1: 1, 2: 0})

# 3. عمل Scaling للأعمدة الثلاثة
features = ['Age', 'Year', 'Nodes']
scaler = StandardScaler()
df[features] = scaler.fit_transform(df[features])

# 4. فصل 20% داتا اختبار للسيرفر المركزي
train_df, test_df = train_test_split(
    df, test_size=0.2, random_state=42, stratify=df['Survival']
)
test_df.to_csv(os.path.join(folder_path, "global_test.csv"), index=False)

# 5. تقسيم الـ 80% الباقية على 3 مستشفيات بـ Pandas مباشرة (لتجنب خطأ NumPy)
n = len(train_df) // 3
h1 = train_df.iloc[:n]
h2 = train_df.iloc[n:2*n]
h3 = train_df.iloc[2*n:]

# 6. حفظ ملف كل مستشفى جوه فولدر fl_project
h1.to_csv(os.path.join(folder_path, "hospital_1.csv"), index=False)
h2.to_csv(os.path.join(folder_path, "hospital_2.csv"), index=False)
h3.to_csv(os.path.join(folder_path, "hospital_3.csv"), index=False)

print("✅ تم التقسيم بنجاح! بص على فولدر fl_project هتلاقي الملفات دي اتعملت:")
print(f"🏥 مستشفى 1 (hospital_1.csv): {len(h1)} مريض")
print(f"🏥 مستشفى 2 (hospital_2.csv): {len(h2)} مريض")
print(f"🏥 مستشفى 3 (hospital_3.csv): {len(h3)} مريض")
print(f"🧪 ملف الاختبار العام (global_test.csv): {len(test_df)} مريض")