
import pandas as pd
from pathlib import Path

file_path = Path(__file__).parent / "dama_lama_synthetic_midterm_2026.csv"
df = pd.read_csv(file_path)

# check data
print(df.shape)
print(df.head())

# columns
print(df.columns)

# data types
print(df.dtypes)

# missing values
print(df.isnull().sum())

# missing percentage
print(df.isnull().mean() * 100)

# duplicates
print(df.duplicated(
    subset=["Facility_Code", "Encounter_Number"]
).sum())

# target
print(df["DAMA_LAMA_Target"].value_counts())

# target percentage
print(df["DAMA_LAMA_Target"].value_counts(normalize=True) * 100)

# statistics
print(df.describe())


# Step 2 - Data Cleaning

# fill missing values
df["Gender"] = df["Gender"].fillna("Unknown")

df["Arrival_Mode"] = df["Arrival_Mode"].fillna("Unknown")

df["Waiting_Time_Minutes"] = df["Waiting_Time_Minutes"].fillna(
    df["Waiting_Time_Minutes"].median()
)

# change date type
df["Register_Date"] = pd.to_datetime(df["Register_Date"])

# check missing values again
print(df.isnull().sum())

# check data types
print(df.dtypes)


# Step 3 - Feature Engineering

# get hour from registration date
df["Hour"] = df["Register_Date"].dt.hour

# get day of week
df["Day"] = df["Register_Date"].dt.dayofweek

# get month
df["Month"] = df["Register_Date"].dt.month

# check new columns
print(df[["Hour", "Day", "Month"]].head())

# select features
features = [
    "Facility_Code",
    "Age",
    "Gender",
    "Triage_Level",
    "Arrival_Mode",
    "Waiting_Time_Minutes",
    "Previous_ED_Visits",
    "Hour",
    "Day",
    "Month"
]

# target
target = "DAMA_LAMA_Target"

X = df[features]
y = df[target]

print(X.shape)
print(y.shape)


# Step 4 - Split Data

from sklearn.model_selection import train_test_split

# split train and test
df_full_train, df_test = train_test_split(
    df, test_size=0.2, random_state=42,
    stratify=df["DAMA_LAMA_Target"]
)

# split train and validation
df_train, df_val = train_test_split(
    df_full_train, test_size=0.25, random_state=42,
    stratify=df_full_train["DAMA_LAMA_Target"]
)

# reset index
df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

print("Train:", df_train.shape)
print("Validation:", df_val.shape)
print("Test:", df_test.shape)


# Step 5 - Logistic Regression

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

# select features
features = [
    "Facility_Code",
    "Age",
    "Gender",
    "Triage_Level",
    "Arrival_Mode",
    "Waiting_Time_Minutes",
    "Previous_ED_Visits",
    "Hour",
    "Day",
    "Month"
]

# target
y_train = df_train["DAMA_LAMA_Target"]
y_val = df_val["DAMA_LAMA_Target"]

# convert data to dictionaries
train_dict = df_train[features].to_dict(orient="records")
val_dict = df_val[features].to_dict(orient="records")

# vectorize
dv = DictVectorizer(sparse=False)

X_train = dv.fit_transform(train_dict)
X_val = dv.transform(val_dict)

# train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# predict probabilities
y_pred = model.predict_proba(X_val)[:, 1]

# evaluate
auc = roc_auc_score(y_val, y_pred)

print("Logistic Regression AUC:", auc)


# Step 6 - Decision Tree

from sklearn.tree import DecisionTreeClassifier

# train decision tree
dt = DecisionTreeClassifier(
    max_depth=5,
    min_samples_leaf=20,
    random_state=42
)

dt.fit(X_train, y_train)

# prediction
y_pred_dt = dt.predict_proba(X_val)[:, 1]

# calculate AUC
auc_dt = roc_auc_score(y_val, y_pred_dt)

print("Decision Tree AUC:", auc_dt)

# compare models
print("Logistic Regression:", auc)
print("Decision Tree:", auc_dt)


# Step 7 - Random Forest

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train, y_train)

y_pred_rf = rf.predict_proba(X_val)[:, 1]

auc_rf = roc_auc_score(y_val, y_pred_rf)

print("Random Forest AUC:", auc_rf)

# compare models
print("\nModel Results")
print("Logistic Regression:", round(auc, 4))
print("Decision Tree:", round(auc_dt, 4))
print("Random Forest:", round(auc_rf, 4))


# Step 8 - Random Forest Tuning

from sklearn.ensemble import RandomForestClassifier

for depth in [5, 10, 15, 20]:

    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=depth,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1
    )

    rf.fit(X_train, y_train)

    y_pred = rf.predict_proba(X_val)[:, 1]

    auc_score = roc_auc_score(y_val, y_pred)

    print("Depth:", depth, "AUC:", round(auc_score, 4))

# Step 9 - Final Model

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

# load original data again
df = pd.read_csv(file_path)

# date features
df["Register_Date"] = pd.to_datetime(df["Register_Date"])
df["Hour"] = df["Register_Date"].dt.hour
df["Day"] = df["Register_Date"].dt.dayofweek
df["Month"] = df["Register_Date"].dt.month

# split data
df_full_train, df_test = train_test_split(
    df, test_size=0.2, random_state=42,
    stratify=df["DAMA_LAMA_Target"]
)

df_train, df_val = train_test_split(
    df_full_train, test_size=0.25, random_state=42,
    stratify=df_full_train["DAMA_LAMA_Target"]
)

# fill missing values
median_wait = df_train["Waiting_Time_Minutes"].median()

for data in [df_train, df_val, df_test]:
    data["Gender"] = data["Gender"].fillna("Unknown")
    data["Arrival_Mode"] = data["Arrival_Mode"].fillna("Unknown")
    data["Waiting_Time_Minutes"] = data["Waiting_Time_Minutes"].fillna(median_wait)

# features
features = [
    "Facility_Code",
    "Age",
    "Gender",
    "Triage_Level",
    "Arrival_Mode",
    "Waiting_Time_Minutes",
    "Previous_ED_Visits",
    "Hour",
    "Day",
    "Month"
]

# combine train and validation
df_full_train = pd.concat([df_train, df_val])

y_full_train = df_full_train["DAMA_LAMA_Target"]
y_test = df_test["DAMA_LAMA_Target"]

# vectorizer
dv = DictVectorizer(sparse=False)

X_full_train = dv.fit_transform(
    df_full_train[features].to_dict(orient="records")
)

X_test = dv.transform(
    df_test[features].to_dict(orient="records")
)

# final model
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1
)

model.fit(X_full_train, y_full_train)

# test predictions
y_pred = model.predict_proba(X_test)[:, 1]

# test AUC
auc_test = roc_auc_score(y_test, y_pred)

print("Final Test AUC:", round(auc_test, 4))

# Step 10 - Save Model

import pickle
from pathlib import Path

model_file = Path(__file__).parent / "model.bin"

with open(model_file, "wb") as f:
    pickle.dump((dv, model, median_wait), f)

print("Model saved successfully")
