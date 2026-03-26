import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor


df = pd.read_csv("Medicalpremium.csv")


df = df.drop("Age", axis=1)

for col in df.columns:
    if df[col].dtype == 'object': 
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

X = df.drop("PremiumPrice", axis=1)
y = df["PremiumPrice"].values

scaler_x = StandardScaler()
X = scaler_x.fit_transform(X)

y_log = np.log1p(y)  


X_train, X_test, y_train, y_test = train_test_split(X, y_log, test_size=0.2, random_state=42)


model = RandomForestRegressor(random_state=42, n_estimators=200)
model.fit(X_train, y_train)


y_pred_log = model.predict(X_test)


y_pred = np.expm1(y_pred_log)  
y_test_original = np.expm1(y_test)


accuracy = r2_score(y_test_original, y_pred) * 100
print("R2 Score:", accuracy)


plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.hist(y, bins=15, color='skyblue', edgecolor='black')
plt.title("Original y (PremiumPrice)")

plt.subplot(1,2,2)
plt.hist(y_log, bins=15, color='salmon', edgecolor='black')
plt.title("Log-Transformed y (log1p)")

plt.tight_layout()
plt.show()



