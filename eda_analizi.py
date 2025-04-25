import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import numpy as np
import missingno as msno
from sklearn.preprocessing import StandardScaler
import plotly.express as px
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, AdaBoostClassifier
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

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

def  Numeric_Veriler_için_Görselleştirme_3_Veri_Seti_için_Kde_Grafik(test_df):
    genel_değişkenler_isimleri = test_df.select_dtypes(include=["int64", "float64"]).columns
    for col in genel_değişkenler_isimleri:
        sns.kdeplot(test_df[col], label='test_df', fill=True, alpha=0.3)
        sns.kdeplot(org_df[col], label='org_df', fill=True, alpha=0.3)
        sns.kdeplot(train_df[col], label='train_df', fill=True, alpha=0.3)
        plt.title("KDE ile Üst Üste Dağılım")
        plt.legend()
        plt.grid()
        plt.show(block=True)
Numeric_Veriler_için_Görselleştirme_3_Veri_Seti_için_Kde_Grafik(test_df)
""" 
Testde bir tek target değişken yok
Test veri seti kolonları ile Target dahil olmadan görselleştirelim 
"""
# Numeric veriler için görselleştirme 3 veri seti için kde grafik
genel_değişkenler = test_df.select_dtypes(include=["int64", "float64"]).columns
# numeric veriler için kdeplot sns ile kodu
"""
sns.kdeplot(test_df['person_age'], label='test_df', fill=True, alpha=0.3)
sns.kdeplot(org_df['person_age'], label='org_df', fill=True, alpha=0.3)
sns.kdeplot(train_df['person_age'], label='train_df', fill=True, alpha=0.3)
plt.title("KDE ile Üst Üste Dağılım")
plt.legend()
plt.show(block=True)
"""
# Projeye eklenmek isterse kde benzeri hist express kütüphanesi kodu
"""
test_df['source'] = 'test_df'
org_df['source'] = 'org_df'
train_df['source'] = 'train_df'

combined_df = pd.concat([test_df, org_df, train_df])
fig = px.histogram(combined_df,
                   x='person_age',
                   color='source',
                   histnorm='density',
                    marginal='rug',
                   opacity=0.5,
                   nbins=50)

fig.update_layout(title_text='Yoğunluk Histogramı (KDE Alternatifi)')
fig.show()

"""

def  Kategorik_Veriler_için_Görselleştirme_3_Veri_Seti_için_countplot_Grafik(test_df):
    genel_değişkenler = test_df.select_dtypes(include=["object"]).columns
    for col in genel_değişkenler:
        sns.countplot(x=test_df[col], label='test_df', fill=True, alpha=0.3)
        sns.countplot(x=org_df[col], label='org_df', fill=True, alpha=0.3)
        sns.countplot(x=train_df[col], label='train_df', fill=True, alpha=0.3)
        plt.title("countplot ile Alt Kategori Üst Üste Frekans Dağılım")
        plt.legend()
        plt.show(block=True)
Kategorik_Veriler_için_Görselleştirme_3_Veri_Seti_için_countplot_Grafik(test_df)

def  Hedef_Değişken_için_Görselleştirme_org_train_Seti_için_countplot_Grafik():
        sns.countplot(x=org_df["loan_status"], label='org_df', fill=True, alpha=0.3)
        sns.countplot(x=train_df["loan_status"], label='train_df', fill=True, alpha=0.3)
        plt.title("countplot ile Hedef Değişken Alt Kategori Üst Üste Frekans Dağılım")
        plt.legend()
        plt.show(block=True)
Hedef_Değişken_için_Görselleştirme_org_train_Seti_için_countplot_Grafik()

# Train eksiklik yok
train_df.isnull().sum()
# Test eksiklik yok
test_df.isnull().sum()
# org eksiklik
org_df.isnull().sum()
org_df.isnull().sum() / len(org_df)*100
"""
person_emp_length             %2.747000
loan_int_rate                 %9.563856
"""

def Duplicate_Veri_Varmi(org_df,test_df,train_df):
    df_list = [("org_df", org_df), ("test_df", test_df), ("train_df", train_df)]
    for df_name, df in df_list:
        total_rows = df.shape[0]
        unique_rows = df.drop_duplicates().shape[0]
        duplicate_count = total_rows - unique_rows
        print(f"{df_name}: Toplam kayıt = {total_rows}, Benzersiz kayıt = {unique_rows}, Duplicate = {duplicate_count}")
Duplicate_Veri_Varmi(org_df,test_df,train_df)

def outlier_threshold(data, col_name, w1=0.01, w2=0.99):
    q1 = data[col_name].quantile(w1)
    q3 = data[col_name].quantile(w2)
    IQR = q3 - q1
    up = q3 + ( 1.5 * IQR)
    low = q1 - ( 1.5 * IQR)
    return up,low
def check_outlier(data, col_name, w1=0.01, w2=0.99):
    up, low = outlier_threshold(data, col_name, w1, w2)
    if data[(data[col_name]<low) | (data[col_name]>up)][col_name].any(axis=None):
        return True
    else:
        return False


print("####### Orjinal Veri Seti Outlier Var mı #######")
for col in org_df.select_dtypes(include=["int64", "float64"]).columns:
    print(col, check_outlier(org_df, col))

print("####### Test Veri Seti Outlier Var mı #######")
for col in test_df.select_dtypes(include=["int64", "float64"]).columns:
    print(col, check_outlier(test_df, col))

print("####### Train Veri Seti Outlier Var mı #######")
for col in train_df.select_dtypes(include=["int64", "float64"]).columns:
    print(col, check_outlier(train_df, col))

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
Korelasyon_Grafiği_Oluştur(org_df)
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
        high_corr_pairs = corr_filtered.stack().loc[lambda x: x < th]
        print(high_corr_pairs)
Korelasyon_Eşiğine_Göre_Değişkenleri_Sırala(org_df,0.8)
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
Hedef_Değişken_ile_Korelasyon_Grafiği_Oluştur(org_df)
Hedef_Değişken_ile_Korelasyon_Grafiği_Oluştur(train_df)

def Hedef_Değişken_analizi_Kategorik(df):
    cat_cols = df.select_dtypes(include="object")

    for col in cat_cols:
        print(df.groupby(col).agg({"loan_status": ["mean","count"]}))
Hedef_Değişken_analizi_Kategorik(org_df)
Hedef_Değişken_analizi_Kategorik(train_df)

def Hedef_Değişken_analizi_Numeric(df):
    num_cols = df.select_dtypes(exclude="object")

    for col in num_cols:
        print(df.groupby("loan_status").agg({col: ["mean","count"]}))
Hedef_Değişken_analizi_Numeric(org_df)
Hedef_Değişken_analizi_Numeric(train_df)


# Base model veri ön işleme olmadan


# tartışma konusu problemler
test_df["ek_özellik"] = ((test_df["loan_amnt"] / test_df["person_income"]) - test_df["loan_percent_income"])
test_df["ek_özellik"].describe()
test_df[test_df["ek_özellik"] > 0]
test_df[test_df["ek_özellik"] == 0]["ek_özellik"].count()


def loan_percent_sorunu(df):
    df["ek_özellik"] = ((df["loan_amnt"] / df["person_income"]) - df["loan_percent_income"])
    print(df["ek_özellik"].describe())
    print(f"Veri seti boyutu: {len(df)}")
    print(f"0 dan büyük olan değer sayısı: {df[df['ek_özellik'] > 0]['ek_özellik'].count()}")
    print(f"0 dan küçük olan değer sayısı: {df[df['ek_özellik'] < 0]['ek_özellik'].count()}")
    print(f"0'a eşit olan değer sayısı: {df[df['ek_özellik'] == 0]['ek_özellik'].count()}")

loan_percent_sorunu(org_df)
loan_percent_sorunu(train_df)
loan_percent_sorunu(test_df)




