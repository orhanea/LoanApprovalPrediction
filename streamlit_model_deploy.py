import pandas as pd
import joblib
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import precision_score, accuracy_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler

pd.set_option("display.width", 500)
pd.set_option("display.max_columns", None)

def veri_seti_düzenlemeleri():
    def veri_setlerini_okut():
        train_df = pd.read_csv("datasets/train.csv")
        test_df = pd.read_csv("datasets/test.csv")
        org_df = pd.read_csv("datasets/credit_risk_dataset.csv")

        print("#######   Veri Setleri Tanımlandı   #######", end="\n\n")
        return train_df, test_df, org_df

    train_df, test_df, org_df = veri_setlerini_okut()

    def Orjinal_veri_seti_ile_Eşleşmeyen_Kolonları_Kaldır(org_df, test_df, train_df):
        org_df_num_cols = org_df.select_dtypes(include=["int64", "float64"]).columns
        test_df_num_cols = test_df.select_dtypes(include=["int64", "float64"]).columns
        train_df_num_cols = train_df.select_dtypes(include=["int64", "float64"]).columns

        train_df_num_cols.difference(org_df_num_cols)  # Index(['id'], dtype='object')
        org_df_num_cols.difference(train_df_num_cols)

        test_eşleşmeyen_kolon = test_df_num_cols.difference(org_df_num_cols)  # Index(['id'], dtype='object')
        test_df = test_df.drop(columns=test_eşleşmeyen_kolon)

        train_eşleşmeyen_kolon = train_df_num_cols.difference(org_df_num_cols)  # Index(['id'], dtype='object')
        train_df = train_df.drop(columns=train_eşleşmeyen_kolon)

        print("#######   Orjinal veri seti ile eşleşmeyen Kolonlar Kaldırıldı   #######", end="\n\n")
        return test_df, train_df, org_df

    test_df, train_df, org_df = Orjinal_veri_seti_ile_Eşleşmeyen_Kolonları_Kaldır(org_df, test_df, train_df)

    def veri_setini_birleştir_loanpercentincome_değişkeni_düzenle(org_df, test_df, train_df):
        test_df["loan_percent_income"] = (test_df["loan_amnt"] / test_df["person_income"])
        train_df["loan_percent_income"] = (train_df["loan_amnt"] / train_df["person_income"])
        org_df["loan_percent_income"] = (org_df["loan_amnt"] / org_df["person_income"])
        train_df = pd.concat([org_df, train_df], axis=0, ignore_index=True)
        return train_df, test_df

    train_df, test_df = veri_setini_birleştir_loanpercentincome_değişkeni_düzenle(org_df, test_df, train_df)
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
            replace_with_thresholds(train_df, col)

        for col in num_cols:
            replace_with_thresholds(test_df, col)

        train_df["person_emp_length"] = train_df["person_emp_length"].fillna(train_df["person_emp_length"].median())
        train_df["loan_int_rate"] = train_df["loan_int_rate"].fillna(train_df["loan_int_rate"].median())
        return train_df, test_df

    train_df, test_df = outlier_baskılama_eksik_veri_doldurma_duplice_kaldırma(train_df, test_df)

    # Feature Engineering
    train_df["NEW_income_interaction"] = train_df["person_income"] - train_df["loan_percent_income"]
    test_df["NEW_income_interaction"] = test_df["person_income"] - test_df["loan_percent_income"]

    train_df['NEW_yas_borc_farki'] = train_df['loan_amnt'] - train_df['person_age'] * 100
    test_df['NEW_yas_borc_farki'] = test_df['loan_amnt'] - test_df['person_age'] * 100

    train_df['NEW_yıllık_ort_kredi_kullanım'] = train_df['loan_amnt'] / train_df['cb_person_cred_hist_length']
    test_df['NEW_yıllık_ort_kredi_kullanım'] = test_df['loan_amnt'] / test_df['cb_person_cred_hist_length']

    return train_df, test_df
train_df, test_df = veri_seti_düzenlemeleri()


def pipeline(train_df):
    # Kategorik ve sayısal sütunları belirleyin
    cat_cols = list(train_df.select_dtypes(include="object").columns)
    num_cols = list(train_df.select_dtypes(exclude="object").columns)
    num_cols.remove("loan_status")
    # Özellik ve hedef değişkeni ayırma
    X = train_df.drop("loan_status",axis=1)  # Features
    y = train_df["loan_status"]  # Target

    # Veriyi eğitim ve test setlerine ayırma
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(drop='first'), cat_cols),  # OHE işlemi
            ('num', StandardScaler(), num_cols)  # Sayısal veriyi ölçeklendirme
        ])

    best_param = {
        "n_estimators": 315,
        "max_depth": 7,
        "learning_rate": 0.015065799174938749,
        "subsample": 0.857704682550547,
        "colsample_bytree": 0.7543243535845349,
        "gamma": 3.187567345636977,
        "reg_alpha": 3.3882365659037825,
        "reg_lambda": 1.7091551711347388
    }
    # Modeli oluşturun ve eğitim verisiyle eğitin
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', XGBClassifier(**best_param))  # Model parametreleri buraya eklenebilir
    ])

    # Modeli eğitin
    model.fit(X_train,y_train)

    # Modeli kaydedin
    joblib.dump(model, 'streamlit/model_with_preprocessor.pkl')

    # Modeli test verisiyle değerlendirin
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])

    # Değerlendirme metrikleri
    print("\n🔍 Değerlendirme Metrikleri:")
    print(f"Accuracy     : {acc:.4f}")
    print(f"Precision    : {prec:.4f}")
    print(f"Recall       : {rec:.4f}")
    print(f"F1-Score     : {f1:.4f}")
    print(f"ROC AUC      : {roc_auc:.4f}")
pipeline(train_df)


