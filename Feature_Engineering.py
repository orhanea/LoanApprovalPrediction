import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("TkAgg")
import seaborn as sns
import numpy as np
import plotly.express as px

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split,cross_validate
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

def veri_setini_birleştir_loanpercentincome_değişkeni_düzenle(org_df,test_df,train_df):
    test_df["loan_percent_income"] = (test_df["loan_amnt"] / test_df["person_income"])
    train_df["loan_percent_income"] = (train_df["loan_amnt"] / train_df["person_income"])
    org_df["loan_percent_income"] = (org_df["loan_amnt"] / org_df["person_income"])
    train_df = pd.concat([org_df, train_df], axis=0, ignore_index=True)
    return train_df,test_df
train_df, test_df = veri_setini_birleştir_loanpercentincome_değişkeni_düzenle(org_df,test_df,train_df)
del org_df


def outlier_threshold(data, col_name, w1=0.01, w2=0.99):
    q1 = data[col_name].quantile(w1)
    q3 = data[col_name].quantile(w2)
    IQR = q3 - q1
    up = q3 + (1.5 * IQR)
    low = q1 - (1.5 * IQR)
    return up, low
def replace_with_thresholds(data, col_name, w1=0.01, w2=0.99):
    up, low = outlier_threshold(data, col_name, w1, w2)
    data.loc[data[col_name] > up, col_name] = float(up)
    data.loc[data[col_name] < low, col_name] = float(low)
def outlier_baskılama_eksik_veri_doldurma_duplice_kaldırma(train_df, test_df):
    train_df = train_df.drop_duplicates()
    num_cols = list(train_df.select_dtypes(include=["int64", "float64"]).columns)
    num_cols.remove("loan_status")
    for col in num_cols:
        replace_with_thresholds(train_df,col)

    for col in num_cols:
        replace_with_thresholds(test_df,col)

    train_df["person_emp_length"] = train_df["person_emp_length"].fillna(train_df["person_emp_length"].median())
    train_df["loan_int_rate"] = train_df["loan_int_rate"].fillna(train_df["loan_int_rate"].median())
    return train_df,test_df
train_df, test_df = outlier_baskılama_eksik_veri_doldurma_duplice_kaldırma(train_df, test_df)

def Korelasyon_Grafiği_Oluştur(df):
    f,ax = plt.subplots(figsize=[4,4])
    sns.heatmap(df[df.select_dtypes(include=["int64", "float64"]).columns].corr(),
                annot=True,
                fmt=".2f",
                ax=ax,
                cmap="YlGnBu",
                linewidths=5)
    ax.set_title("Correlation Matrix",fontsize=20)
    plt.subplots_adjust(bottom=0.3)
    plt.show()
Korelasyon_Grafiği_Oluştur(test_df)
Korelasyon_Grafiği_Oluştur(train_df)

def Korelasyon_Eşiğine_Göre_Değişkenleri_Sırala(df,th,yön=True):
    corr_matrix = df[df.select_dtypes(include=["int64", "float64"]).columns].corr()
    mask = np.triu(np.ones(corr_matrix.shape), k=0)
    corr_filtered = corr_matrix.where(mask == 0)
    if yön:
        high_corr_pairs = corr_filtered.stack().loc[lambda x: x > th]
        print(high_corr_pairs)
    else:
        low_corr_pairs = corr_filtered.stack().loc[lambda x: x < th]
        print(low_corr_pairs)
Korelasyon_Eşiğine_Göre_Değişkenleri_Sırala(test_df,0.8)
Korelasyon_Eşiğine_Göre_Değişkenleri_Sırala(train_df,0.8)

def Hedef_Değişken_ile_Korelasyon_Grafiği_Oluştur(df):
    num_cols = df.select_dtypes(include=["int64", "float64"]).columns

    df[num_cols].corrwith(df["loan_status"]).sort_values(ascending=False)

    corr_data = df[num_cols].corrwith(df["loan_status"]).to_frame()
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(corr_data,
                annot=True, fmt=".2f",
                ax=ax, cmap="YlGnBu",
                linewidths=5)
    plt.subplots_adjust(left=0.3)
    plt.show()
Hedef_Değişken_ile_Korelasyon_Grafiği_Oluştur(train_df)


def dummies_cat_col(df):
    cat_cols = df.select_dtypes(include="object").columns
    df = pd.get_dummies(data=df,columns=cat_cols,drop_first=True)
    return df
train_df = dummies_cat_col(train_df)
test_df = dummies_cat_col(test_df)

def scale_num_col(train,test):
    num_cols = list(train.select_dtypes(include=["float64", "int64"]).columns)
    num_cols.remove("loan_status")
    scaler = StandardScaler()
    scaler.fit(train[num_cols])
    train[num_cols] = scaler.transform(train[num_cols])
    test[num_cols] = scaler.transform(test[num_cols])
    return train , test
train_df,test_df = scale_num_col(train_df,test_df)

X = train_df.drop("loan_status", axis=1)  # Features
y = train_df["loan_status"]  # Target

cv_results = cross_validate(XGBClassifier(random_state=42),
                            X,
                            y,
                            cv=10,
                            scoring=["roc_auc","accuracy","precision","recall"])

print(f"Recall: {round(cv_results['test_recall'].mean(), 4)} ")
print(f"Precision: {round(cv_results['test_precision'].mean(), 4)} ")
print(f"Accuracy: {round(cv_results['test_accuracy'].mean(), 4)} ")
print(f"Roc_auc: {round(cv_results['test_roc_auc'].mean(), 4)}")


"""
Recall: 0.7296 
Precision: 0.9145 
Accuracy: 0.9426 
Roc_auc: 0.9472
"""