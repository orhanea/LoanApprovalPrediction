import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import numpy as np
import plotly.express as px
import shap

from xgboost import plot_importance
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split,cross_validate
from xgboost import XGBClassifier
from sklearn.metrics import classification_report

matplotlib.use("TkAgg")
pd.set_option("display.width",500)
pd.set_option("display.max_columns",None)

def veri_seti_düzenlemeleri():
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
    train_df.to_csv("düzensiz_merge_train.csv")
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
    return train_df, test_df
train_df, test_df = veri_seti_düzenlemeleri()

"""
Kötü etki eden değişkenler

def rare_encoder(df, col, th_rare):
    değişken_sınıf_yüzdeleri = df[col].value_counts() / df.shape[0]
    rare_indx_list = değişken_sınıf_yüzdeleri[değişken_sınıf_yüzdeleri < th_rare].index
    df[col] = np.where(df[col].isin(rare_indx_list), "Rare", df[col])
rare_encoder(train_df, "loan_grade", 0.1)
##########################################################33
# Kötü başarı metrikleri elde ettiği değişkenler
train_df.drop("NEW_Rent_Risk",inplace=True,axis=1)
train_df["NEW_Rent_Risk"] = train_df["NEW_Rent_Risk"].fillna(0)
train_df["loan_grade"].value_counts() / len(train_df)


train_df.loc[((train_df["person_home_ownership"] == "RENT") | (train_df["person_home_ownership"] == "OWN") ) &
             (train_df["loan_grade"] == "Rare") &
             (train_df["cb_person_default_on_file"] == "Y"),"NEW_Rent_Risk" ] = 1
             
             
train_df["person_income"] = pd.qcut(train_df["person_income"],q=3,labels=["küçük","orta","büyük"])

train_df['NEW_income_to_age'] = train_df['person_income'] / (train_df['person_age'] *1000)
test_df['NEW_income_to_age'] = test_df['person_income'] / (test_df['person_age'] *1000)


#Geliri düşük ancak faizi yüksek olan kişilerin ödeme riski daha fazla olabilir
train_df["NEW_faiz_oranının_gelire_oranı"] = train_df["loan_int_rate"] / train_df["person_income"]
test_df["NEW_faiz_oranının_gelire_oranı"] = test_df["loan_int_rate"] / test_df["person_income"]

#Ne kadar yüksekse kişi o kadar rahat öder.
train_df["gelirin_toplam_kredi_maliyetine_oranı"] = train_df["person_income"] / (train_df["loan_amnt"] * train_df["loan_int_rate"])
test_df["gelirin_toplam_kredi_maliyetine_oranı"] = test_df["person_income"] / (test_df["loan_amnt"] * test_df["loan_int_rate"])
    Recall: 0.7281 
    Precision: 0.9127 
    Accuracy: 0.9421 
    Roc_auc: 0.9477
    
#Kişinin zaman içindeki gelirini ölçer.
train_df["NEW_istihdam_yılı_başına_düşen_gelir"] = train_df["person_income"] / (train_df["person_emp_length"] + 1)
test_df["NEW_istihdam_yılı_başına_düşen_gelir"] = test_df["person_income"] / (test_df["person_emp_length"] + 1)
    Recall: 0.7279 
    Precision: 0.9153 
    Accuracy: 0.9425 
    Roc_auc: 0.9473
    
    
train_df['toplam_faiz'] = train_df['loan_amnt']*train_df['loan_int_rate']/train_df['person_income']
test_df['toplam_faiz'] = test_df['loan_int_rate']*test_df['loan_amnt']/test_df['person_income']    
    Recall: 0.7283 
    Precision: 0.9158 
    Accuracy: 0.9426 
    Roc_auc: 0.9473
    
train_df['NEW_age_length'] = train_df['person_age'] - train_df['cb_person_cred_hist_length']
test_df['NEW_age_length'] = test_df['person_age'] - test_df['cb_person_cred_hist_length']    
    Recall: 0.7292 
    Precision: 0.9152 
    Accuracy: 0.9426 
    Roc_auc: 0.9476
    
train_df['NEW_sadakat_bazlı_kredi_yükü'] = train_df['loan_amnt']/train_df['person_income']*train_df['cb_person_cred_hist_length']
test_df['NEW_sadakat_bazlı_kredi_yükü'] = test_df['loan_amnt']/test_df['person_income']*test_df['cb_person_cred_hist_length']
    Recall: 0.7263 
    Precision: 0.9159 
    Accuracy: 0.9423 
    Roc_auc: 0.9463

"""
##########################################################33
# güzel bir etki yarattı modelimde
train_df["NEW_income_interaction"] = train_df["person_income"] - train_df["loan_percent_income"]
test_df["NEW_income_interaction"] = test_df["person_income"] - test_df["loan_percent_income"]

train_df['NEW_yas_borc_farki'] = train_df['loan_amnt'] - train_df['person_age'] * 100
test_df['NEW_yas_borc_farki'] = test_df['loan_amnt'] - test_df['person_age'] * 100

train_df['NEW_yıllık_ort_kredi_kullanım'] = train_df['loan_amnt']/train_df['cb_person_cred_hist_length']
test_df['NEW_yıllık_ort_kredi_kullanım'] = test_df['loan_amnt']/test_df['cb_person_cred_hist_length']

def model_oluştur(train_df,test_df,cv=5,feature_importence=False,shap_sum_plot=False):
    def dummies_cat_col(df):
        cat_cols = df.select_dtypes(include=["object","category"]).columns
        df = pd.get_dummies(data=df,columns=cat_cols,drop_first=True)
        return df
    train_df = dummies_cat_col(train_df)
    test_df = dummies_cat_col(test_df)

    def scale_num_col(train,test):
        num_cols = list(train.select_dtypes(include=["float64", "int64"]).columns)
        num_cols.remove("loan_status")
        #num_cols.remove("NEW_Rent_Risk")
        scaler = StandardScaler()
        scaler.fit(train[num_cols])
        train[num_cols] = scaler.transform(train[num_cols])
        test[num_cols] = scaler.transform(test[num_cols])
        return train , test
    train_df,test_df = scale_num_col(train_df,test_df)

    X = train_df.drop("loan_status", axis=1)  # Features
    y = train_df["loan_status"]  # Target

    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    model = XGBClassifier(random_state=42)
    cv_results = cross_validate(model,
                                X_train,
                                y_train,
                                cv=cv,
                                scoring=["roc_auc","accuracy","precision","recall"])
    print(f"{cv} Katli Capraz Doğrulama sonucu Başarı metriklerimiz",end="\n\n")
    print(f"Recall: {round(cv_results['test_recall'].mean(), 4)} ")
    print(f"Precision: {round(cv_results['test_precision'].mean(), 4)} ")
    print(f"Accuracy: {round(cv_results['test_accuracy'].mean(), 4)} ")
    print(f"Roc_auc: {round(cv_results['test_roc_auc'].mean(), 4)}")
    model = XGBClassifier(random_state=42)
    model.fit(X, y)
    if feature_importence:
        plot_importance(model, importance_type='weight', max_num_features=10)
        plt.show()

    if shap_sum_plot:
        print("Shap grafiği oluşturuluyor.")
        explainer = shap.Explainer(model)
        shap_values = explainer(X)
        shap.summary_plot(shap_values, X)
        print("Shap grafiği oluşturuldu.")
model_oluştur(train_df,test_df,shap_sum_plot=True,feature_importence=True)


"""
Recall: 0.7296 
Precision: 0.9145 
Accuracy: 0.9426 
Roc_auc: 0.9472
"""

"""
# güzel bir etki yarattı modelimde
train_df["NEW_income_interaction"]
Recall: 0.7299 
Precision: 0.9149 
Accuracy: 0.9427 
Roc_auc: 0.9484
"""

"""
train_df[NEW_yas_borc_farki]
Recall: 0.73 
Precision: 0.9152 
Accuracy: 0.9428 
Roc_auc: 0.949
"""

"""
train_df[NEW_yıllık_ort_kredi_kullanım]
Recall: 0.7288 
Precision: 0.9163 
Accuracy: 0.9427 
Roc_auc: 0.9477
"""





"""
Shap grafiği yorumu
person_income = başvuranın yıllık geliri azaldıkça biz kredisini onaylama eğilimindeyiz
loan_int_rate = krediye uygulanan faiz oranı yükseldikçe krediyi onaylama eğilimindeyiz
loan_percent_income = başvuranın yıllık gelirinin ne kadarı kredilere gideği miktar arttıkça kredi onaylama eğilimindeyiz
person_home_ownership_RENT = Kişinin ev sahipliği kirada ise biz kişinin kredisini onaylama eğilimindeyiz
person_home_ownership_Own = Kişi ev sahibi ise kredisini onaylamama eğilimindeyiz
loan_grade_D,loan_grade_E,loan_grade_F = kişinin kredi risk kategorisi D,E,F ise onaylama eğilimindeyiz
cb_person_default_on_file_Y = Kişi temerrüde düşmüş ise kredisini onaylama eğilimindeyiz
"""