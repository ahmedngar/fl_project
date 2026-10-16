# 🏥 Federated Learning for Healthcare Survival Prediction

An end-to-end **Privacy-Preserving Machine Learning** mini-project that demonstrates **Federated Learning (FedAvg)** across multiple hospitals without sharing raw patient data.

---

## 📌 Project Overview & Problem Statement
In the healthcare sector, hospitals cannot directly share sensitive patient records due to strict data privacy regulations. However, training a machine learning model on a single hospital's small dataset often yields suboptimal accuracy.

**The Solution:** This project implements a decentralized **Federated Learning** pipeline where **3 isolated hospitals** train local models on their private data and share **only the model weights (gradients)** with a central server. The server aggregates these weights using the **Federated Averaging (`FedAvg`)** algorithm to build a robust global model.

---

## 📊 Dataset: Haberman's Survival Data
The dataset contains cases from a study conducted on the survival of patients who had undergone surgery for breast cancer:
* **`Age`**: Age of patient at the time of operation.
* **`Year`**: Patient's year of operation (1958 - 1970).
* **`Nodes`**: Number of positive axillary nodes detected.
* **`Survival` (Target)**: `1` = Patient survived 5 years or longer, `0` = Patient died within 5 years.

---

## 📂 Repository Structure
```text
fl_project/
│
├── 1_prepare_data.py       # Data preprocessing, scaling, and partitioning across 3 hospitals
├── 2_train_federated.py    # Federated Averaging (FedAvg) training engine & evaluation
├── 3_app.py                # Interactive Streamlit web application for prediction
├── federated_model.pkl     # Exported global federated model artifact
├── haberman.csv            # Original dataset
├── hospital_1.csv          # Local training data for Hospital 1
├── hospital_2.csv          # Local training data for Hospital 2
├── hospital_3.csv          #
