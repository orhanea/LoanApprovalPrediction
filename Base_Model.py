import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import numpy as np
import missingno as msno
from sklearn.preprocessing import StandardScaler
import plotly.express as px
from sklearn.model_selection import train_test_split,cross_validate
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report

matplotlib.use("TkAgg")
pd.set_option("display.width",500)
pd.set_option("display.max_columns",None)

def veri_setlerini_okut():
    train_df = pd.read_csv("datasets/train.csv")
    test_df = pd.read_csv("datasets/test.csv")
    org_df = pd.read_csv("datasets/credit_risk_dataset.csv")

    print("#######   Veri Setleri Tanımlandı   #######",end="\n\n")
    return train_df,test_df,org_df
train_df,test_df,org_df = veri_setlerini_okut()

def Orjinal_veri_seti_ile_Eşleşmeyen_Kolonları_Kaldır(org_df,test_df,train_df):
    org_df_num_cols = org_df.select_dtypes(include=["int64", "float64"]).columns
    test_df_num_cols = test_df.select_dtypes(include=["int64", "float64"]).columns
    train_df_num_cols = train_df.select_dtypes(include=["int64", "float64"]).columns

    train_df_num_cols.difference(org_df_num_cols) # Index(['id'], dtype='object')

    org_df_num_cols.difference(train_df_num_cols)

    test_eşleşmeyen_kolon = test_df_num_cols.difference(org_df_num_cols) #  Index(['id'], dtype='object')
    test_df = test_df.drop(columns=test_eşleşmeyen_kolon)

    train_eşleşmeyen_kolon = train_df_num_cols.difference(org_df_num_cols) # Index(['id'], dtype='object')
    train_df = train_df.drop(columns=train_eşleşmeyen_kolon)

    print("#######   Orjinal veri seti ile eşleşmeyen Kolonlar Kaldırıldı   #######", end="\n\n")
    return test_df,train_df,org_df
test_df,train_df,org_df = Orjinal_veri_seti_ile_Eşleşmeyen_Kolonları_Kaldır(org_df,test_df,train_df)

def dummies_cat_col(df):
    cat_cols = df.select_dtypes(include="object").columns
    df = pd.get_dummies(data=df,columns=cat_cols,drop_first=True)
    return df
org_df = dummies_cat_col(org_df)
train_df = dummies_cat_col(train_df)
test_df = dummies_cat_col(test_df)

def scale_num_col(df,test):
    num_cols = list(org_df.select_dtypes(include=["float64", "int64"]).columns)
    num_cols.remove("loan_status")
    scaler = StandardScaler()
    scaler.fit(df[num_cols])
    df[num_cols] = scaler.transform(df[num_cols])
    test[num_cols] = scaler.transform(test[num_cols])
    return df , test
org_df,test_df = scale_num_col(org_df,test_df)
train_df,test_df = scale_num_col(train_df,test_df)

# Base Model Train
X = train_df.drop("loan_status", axis=1)  # Features
y = train_df["loan_status"]  # Target

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

classifiers = [('RF', RandomForestClassifier(n_estimators=100, random_state=42)),
               ('XGBoost', XGBClassifier(missing=np.nan, random_state=42))]

for name, classifier in classifiers:
    print(f"##### {name} #####")
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_valid)
    print(classification_report(y_valid, y_pred))

# Cross validasyonu
for name, classifier in classifiers:
        cv_results = cross_validate(classifier, X, y, cv=5, scoring=["roc_auc","accuracy","precision","recall"])
        print(f"Recall: {round(cv_results['test_recall'].mean(), 4)} ({name}) ")
        print(f"Precision: {round(cv_results['test_precision'].mean(), 4)} ({name}) ")
        print(f"Accuracy: {round(cv_results['test_accuracy'].mean(), 4)} ({name}) ")
        print(f"Roc_auc: {round(cv_results['test_roc_auc'].mean(), 4)} ({name}) ")

##################################################################################
# train ve org birleşimi
train_df = pd.concat([org_df, train_df], axis=0, ignore_index=True)

X = train_df.drop("loan_status", axis=1)  # Features
y = train_df["loan_status"]  # Target

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

classifiers = [('RF', RandomForestClassifier(n_estimators=100, random_state=42)),
               ('XGBoost', XGBClassifier(missing=np.nan, random_state=42))]


for name, classifier in classifiers:
    print(f"##### {name} #####")
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_valid)
    print(classification_report(y_valid, y_pred))

for name, classifier in classifiers:
        cv_results = cross_validate(classifier, X_train, y_train ,cv=5, scoring=["roc_auc","accuracy","precision","recall"])
        print(f"Recall: {round(cv_results['test_recall'].mean(), 4)} ({name}) ")
        print(f"Precision: {round(cv_results['test_precision'].mean(), 4)} ({name}) ")
        print(f"Accuracy: {round(cv_results['test_accuracy'].mean(), 4)} ({name}) ")
        print(f"Roc_auc: {round(cv_results['test_roc_auc'].mean(), 4)} ({name}) ")

