# Cybersecurity Intrusion Detection System (IDS) 🔐🧠

This project implements a Machine Learning-based Intrusion Detection System (IDS) for real time classification of network traffic as either **normal** or **malicious**. It features a clean and interactive **Streamlit** web interface and uses a trained **XGBoost** model to achieve high classification performance.

---

🚀 Features

- ✅ Machine Learning-based classification using XGBoost
- ✅ Real-time traffic prediction via interactive Streamlit UI
- ✅ Simple, responsive, and easy to use frontend
- ✅ High classification accuracy (achieved >99.9% on test set)
- ✅ Ready to deploy structure for academic demos or PoC projects

---

📁 Project Structure

| File              | Description                                          |
|-------------------|------------------------------------------------------|
| `app.py`          | Main application file containing Streamlit interface |
| `xgb_model.pkl`   | Trained XGBoost model saved in pickle format         |
| `requirements.txt`| Python dependencies for local installation           |
| `README.md`       | Project documentation                                |

---

🛠️ Installation & Usage

1. Clone the repository:
```bash
git clone https://github.com/DicleCeylann/cybersecurity-ids.git
cd cybersecurity-ids

2. Install the required packages:
 pip install -r requirements.txt

3. Run the application:
 streamlit run app.py

---

🧠 Model Detail

Algorithm: XGBoost (Gradient Boosted Trees)
Task: Binary classification (normal vs. malicious)
Training Accuracy: ~99.9%
Features Used: (List your input features if relevant)
Note: You can retrain the model on your own dataset or extend the current pipeline with preprocessing and feature engineering.

---

👩‍💻 Author

Dicle Ceylan
B.Sc. in Statistics, Yildiz Technical University (Class of 2026)
Focus areas: Cybersecurity, AI, IoT Security
🌐 GitHub: github.com/DicleCeylann
💼 LinkedIn: linkedin.com/in/dicle-ceylan

---

📄 License

This project is open-source and available for academic, educational, and research use.
If you use or modify this code, please provide proper attribution.
