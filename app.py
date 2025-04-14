import pandas as pd

import pickle


with open('/Users/dicleceylan/Desktop/xgb_model.pkl', 'rb') as file:
    model = pickle.load(file)


df_train = pd.read_csv('/Users/dicleceylan/Desktop/UNSW_NB15_training-set.csv')
df_test = pd.read_csv('/Users/dicleceylan/Desktop/UNSW_NB15_testing-set.csv')

print("Training Set")
print(df_train.head())

print("\nTesting Set")
print(df_test.head())

print(df_train.isnull().sum().sort_values(ascending=False))


categorical_cols = df_train.select_dtypes(include=['object']).columns
print(categorical_cols)
from sklearn.preprocessing import LabelEncoder


le = LabelEncoder()


combined = pd.concat([df_train, df_test], axis=0)


categorical_cols = ['proto', 'service', 'state', 'attack_cat']


for col in categorical_cols:
    combined[col] = combined[col].astype(str)


for col in categorical_cols:
    combined[col] = le.fit_transform(combined[col])


df_train = combined.iloc[:len(df_train)]
df_test = combined.iloc[len(df_train):]

print("Tüm kategorik değişkenler düzgünce sayısala çevrildi!")


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


X_train = df_train.drop(['id', 'label'], axis=1) 
y_train = df_train['label']

X_test = df_test.drop(['id', 'label'], axis=1)
y_test = df_test['label']

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)


y_pred = rf.predict(X_test)


print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))


import matplotlib.pyplot as plt
import pandas as pd


importances = rf.feature_importances_
features = X_train.columns
importance_df = pd.DataFrame({'Feature': features, 'Importance': importances})


importance_df = importance_df.sort_values(by='Importance', ascending=False)


plt.figure(figsize=(10,6))
plt.barh(importance_df['Feature'][:10][::-1], importance_df['Importance'][:10][::-1])
plt.xlabel('Feature Importance')
plt.title('Top 10 Important Features (Random Forest)')
plt.show()




import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report


xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
xgb_model.fit(X_train, y_train)


y_pred_xgb = xgb_model.predict(X_test)


print("XGBoost Accuracy:", accuracy_score(y_test, y_pred_xgb))
print("\nXGBoost Classification Report:\n", classification_report(y_test, y_pred_xgb))

import streamlit as st
import numpy as np

st.title('Cybersecurity Intrusion Detection System')


sbytes = st.number_input('Gönderilen Bayt (sbytes)', min_value=0)
dbytes = st.number_input('Alınan Bayt (dbytes)', min_value=0)
sload = st.number_input('Gönderici Trafik Yükü (sload)', min_value=0.0)
sttl = st.number_input('Source TTL (sttl)', min_value=0)
ct_state_ttl = st.number_input('Bağlantı Durum-TTL Sayısı (ct_state_ttl)', min_value=0)


if st.button('Tahmin Yap'):
    input_data = np.array([[sbytes, dbytes, sload, sttl, ct_state_ttl]])
    prediction = model.predict(input_data)[0]  # Şimdilik rastgele
    if prediction == 0:
        st.success('Bu trafik NORMAL ✅')
    else:
        st.error('Bu trafik SALDIRI içeriyor 🚨')



import pickle


with open('/Users/dicleceylan/Desktop/xgb_model.pkl', 'rb') as file:
    model = pickle.load(file)




