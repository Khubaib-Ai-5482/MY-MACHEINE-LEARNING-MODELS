import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler

# ---------- Load Dataset ----------
df = pd.read_csv("used_cars.csv")

# ---------- Encode categorical ----------
cat_data = df.select_dtypes(include="object")
le = LabelEncoder()
for col in cat_data:
    df[col] = le.fit_transform(df[col])

# ---------- EDA ----------
sns.boxplot(data=df)
plt.show()
sns.histplot(df, kde=True, color="skyblue")
plt.show()
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.show()

# ---------- Create bins for sampling ----------
df['price_bin'] = pd.qcut(df['price'], q=3, labels=[0,1,2])  # 3 bins

X = df.drop(['price','price_bin'], axis=1)
y_bin = df['price_bin']

X_train, X_test, y_train_bin, y_test_bin = train_test_split(
    X, y_bin, test_size=0.2, random_state=42, stratify=y_bin
)

y_train_cont = df.loc[y_train_bin.index, 'price']
y_test_cont = df.loc[y_test_bin.index, 'price']

# ---------- Random Undersampling ----------
rus = RandomUnderSampler(random_state=42)
X_rus, y_rus_bin = rus.fit_resample(X_train, y_train_bin)
y_rus_cont = y_train_cont.loc[X_rus.index]

# ---------- Random Oversampling ----------
ros = RandomOverSampler(random_state=42)
X_ros, y_ros_bin = ros.fit_resample(X_train, y_train_bin)
y_ros_cont = y_train_cont.loc[X_ros.index]

# ---------- SMOTE ----------
smote = SMOTE(random_state=42)
X_sm, y_sm_bin = smote.fit_resample(X_train, y_train_bin)
y_sm_cont = y_train_cont.loc[X_sm.index]

# ---------- Train Decision Tree Regressor ----------
model = DecisionTreeRegressor(random_state=42)

# Original
model.fit(X_train, y_train_cont)
y_pred = model.predict(X_test)
print("Original R2:", r2_score(y_test_cont, y_pred))

# Undersampled
model.fit(X_rus, y_rus_cont)
y_pred = model.predict(X_test)
print("Undersampled R2:", r2_score(y_test_cont, y_pred))

# Oversampled
model.fit(X_ros, y_ros_cont)
y_pred = model.predict(X_test)
print("Oversampled R2:", r2_score(y_test_cont, y_pred))

# SMOTE
model.fit(X_sm, y_sm_cont)
y_pred = model.predict(X_test)
print("SMOTE R2:", r2_score(y_test_cont, y_pred))

