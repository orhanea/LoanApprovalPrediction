import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import shap
import optuna
import joblib

from sklearn.metrics import precision_score, accuracy_score, recall_score, f1_score,roc_auc_score
from xgboost import plot_importance
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_validate ,cross_val_score
from xgboost import XGBClassifier

matplotlib.use("TkAgg")
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

def model_oluştur(train_df, test_df, cv=10, feature_importence=False, shap_sum_plot=False):
    def dummies_cat_col(df):
        cat_cols = df.select_dtypes(include=["object", "category"]).columns
        df = pd.get_dummies(data=df, columns=cat_cols, drop_first=True)
        return df

    train_df = dummies_cat_col(train_df)
    test_df = dummies_cat_col(test_df)

    def scale_num_col(train, test):
        num_cols = list(train.select_dtypes(include=["float64", "int64"]).columns)
        num_cols.remove("loan_status")
        # num_cols.remove("NEW_Rent_Risk")
        scaler = StandardScaler()
        scaler.fit(train[num_cols])
        train[num_cols] = scaler.transform(train[num_cols])
        test[num_cols] = scaler.transform(test[num_cols])
        return train, test

    train_df, test_df = scale_num_col(train_df, test_df)

    X = train_df.drop("loan_status", axis=1)  # Features
    y = train_df["loan_status"]  # Target

    model = XGBClassifier(random_state=42)
    cv_results = cross_validate(model,
                                X,
                                y,
                                cv=cv,
                                scoring=["roc_auc", "accuracy", "precision", "recall"])
    print(f"{cv} Katli Capraz Doğrulama sonucu Başarı metriklerimiz", end="\n\n")
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
model_oluştur(train_df, test_df, shap_sum_plot=True, feature_importence=False)

def HyperParametreAnalysis(train_df, test_df,trial=15):
    def dummies_cat_col(df):
        cat_cols = df.select_dtypes(include=["object", "category"]).columns
        df = pd.get_dummies(data=df, columns=cat_cols, drop_first=True)
        return df
    train_df = dummies_cat_col(train_df)
    test_df = dummies_cat_col(test_df)

    def scale_num_col(train, test):
        num_cols = list(train.select_dtypes(include=["float64", "int64"]).columns)
        num_cols.remove("loan_status")
        scaler = StandardScaler()
        scaler.fit(train[num_cols])
        train[num_cols] = scaler.transform(train[num_cols])
        test[num_cols] = scaler.transform(test[num_cols])
        return train, test
    train_df, test_df = scale_num_col(train_df, test_df)

    X = train_df.drop("loan_status", axis=1)  # Features
    y = train_df["loan_status"]  # Target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    """
        # 1. n_estimators:
        # Açıklama: Modeldeki ağaç sayısını belirler.
        # Aralık: 50 ile 500 arasında bir değer önerilir.
        # Etkisi: Bu parametre, modelin kaç adet karar ağacı oluşturacağını belirtir. Çok yüksek değerler aşırı öğrenmeye 
        (overfitting) yol açabilirken, düşük değerler modelin yeterince iyi öğrenmesini engelleyebilir.

        # 2. max_depth:
        # Açıklama: Her karar ağacının derinliğini belirler.
        # Aralık: 3 ile 15 arasında bir değer önerilir.
        # Etkisi: Bu parametre, her bir ağacın ne kadar derinleşebileceğini sınırlar. Derin ağaçlar, daha karmaşık modeller 
        oluşturabilir, ancak aşırı derin ağaçlar aşırı öğrenmeye neden olabilir.
        #
        # 3. learning_rate:
        # Açıklama: Modelin öğrenme hızını belirler.
        # Aralık: 0.01 ile 0.3 arasında bir değer önerilir.
        # Etkisi: Öğrenme hızı, modelin her iterasyonda ne kadar öğrenmesi gerektiğini belirler. Düşük öğrenme hızı genellikle 
        daha doğru sonuçlar verir, ancak eğitim süresi daha uzun olur. Yüksek öğrenme hızı ise daha hızlı sonuç verir, ancak 
        düşük doğrulukla sonuçlanabilir.

        # 4. subsample:
        # Açıklama: Modelin her iterasyonunda kullanılan veri örneklerinin oranını belirler.
        # Aralık: 0.5 ile 1.0 arasında bir değer önerilir.
        # Etkisi: Subsample oranı, her karar ağacının yalnızca rastgele seçilen alt kümesini kullanarak eğitilmesini sağlar. 
        Bu, modelin daha genel olmasını sağlar ve aşırı öğrenmeyi (overfitting) azaltabilir.

        # 5. colsample_bytree:
        # Açıklama: Her karar ağacının eğitiminde kullanılacak özelliklerin oranını belirler.
        # Aralık: 0.5 ile 1.0 arasında bir değer önerilir.
        # Etkisi: Bu parametre, her bir karar ağacında kaç adet özellik kullanılacağını belirler. Düşük değerler daha genel 
        odeller oluşturur, ancak çok düşük değerler de modelin performansını düşürebilir.

        # 6. gamma:
        # Açıklama: Ağaçların bölünmesini kontrol eden bir düzenleme parametresi.
        # Aralık: 0 ile 5 arasında bir değer önerilir.
        # Etkisi: Gamma, ağaç bölünmelerini daha katı hale getirir. Daha yüksek gamma, daha az ağacın bölünmesine neden olur, 
        bu da modelin daha basit ve daha az aşırı öğrenmiş (overfitting) olmasına yol açabilir.

        # 7. reg_alpha:
        # Açıklama: L1 düzenleme parametresi.
        # Aralık: 0 ile 5 arasında bir değer önerilir.
        # Etkisi: L1 düzenlemesi, ağırlıkları sıfıra yakınlaştırarak bazı özelliklerin (features) etkisini ortadan kaldırır. 
        Bu, modelin daha basit olmasını sağlar ve aşırı öğrenmeyi (overfitting) engellemeye yardımcı olabilir.

        # 8. reg_lambda:
        # Açıklama: L2 düzenleme parametresi.
        # Aralık: 0 ile 5 arasında bir değer önerilir.
        # Etkisi: L2 düzenlemesi, ağırlıkları küçülterek büyük ağırlıkların önüne geçer. Bu da modelin daha düzgün ve genelleştirilebilir 
        olmasını sağlar.

        # 9. eval_metric:
        # Açıklama: Değerlendirme metriği belirler.
        # Değer: "logloss"
        # Etkisi: Bu parametre, modelin eğitim sırasında hangi metrikle değerlendirileceğini belirtir. logloss, 
        # doğruluğa odaklanmaktan daha iyi bir performans ölçütüdür, özellikle sınıflandırma problemleri için kullanılır. 
            Bu, modelin logaritmik kaybını ölçer ve modelin doğruluğu hakkında daha hassas bilgi verir.
    """
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 15),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "gamma": trial.suggest_float("gamma", 0, 5),
            "reg_alpha": trial.suggest_float("reg_alpha", 0, 5),
            "reg_lambda": trial.suggest_float("reg_lambda", 0, 5),
            "eval_metric": "logloss"
        }

        model = XGBClassifier(**params)

        # Precision skoru optimize et

        precision = cross_val_score(
            model,
            X_train,
            y_train,
            cv=3,
            scoring='precision'  # Burada doğrudan precision kullanıyoruz
        ).mean()

        return precision

    # Optimizasyon başlat
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=trial)

    # En iyi parametrelerle model
    print("\n📌 En iyi parametreler:", study.best_params)
    best_model = XGBClassifier(**study.best_params)
    best_model.fit(X_train, y_train)
    y_pred = best_model.predict(X_test)

    # Değerlendirme metrikleri
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, best_model.predict_proba(X_test)[:, 1])

    print("\n🔍 Değerlendirme Metrikleri:")
    print(f"Recall       : {rec:.4f}")
    print(f"Accuracy     : {acc:.4f}")
    print(f"Precision    : {prec:.4f}")
    print(f"ROC AUC      : {roc_auc:.4f}")

    return best_model, train_df ,test_df
best_model, train_df ,test_df = HyperParametreAnalysis(train_df,test_df)

# Tahmin Sonuçlarını Yerleştirme DataFreme'a
y_pred = best_model.predict(test_df)
test_df["pred"] = y_pred

# Örneklem oluşturalım Tahmin edelim
test_df.iloc[0].values
best_model.predict([
        23, 69000.0, 'RENT', 3.0,
       'HOMEIMPROVEMENT', 'F', 25000, 15.76,
        0.36231884057971014, 'N',2.0,
        68999.63768115942,22700,
        12500.0])
"""
📌 En iyi parametreler: 
{'n_estimators': 315, 
'max_depth': 7, 
'learning_rate': 0.015065799174938749, 
'subsample': 0.857704682550547, 
'colsample_bytree': 0.7543243535845349, 
'gamma': 3.187567345636977, 
'reg_alpha': 3.3882365659037825, 
'reg_lambda': 1.7091551711347388}


🔍 Değerlendirme Metrikleri:
Accuracy     : 0.9416
Precision    : 0.9525
Recall       : 0.6942
F1-Score     : 0.8031
"""

# Ekip arkadaşları için parametre kombinasyonları
train_df, test_df = veri_seti_düzenlemeleri()
params_1 = {
    "n_estimators": 315,
    "max_depth": 7,
    "learning_rate": 0.015065799174938749,
    "subsample": 0.857704682550547,
    "colsample_bytree": 0.7543243535845349,
    "gamma": 3.187567345636977,
    "reg_alpha": 3.3882365659037825,
    "reg_lambda": 1.7091551711347388
}
params_2 = {'n_estimators': 300, 'max_depth': 7,
          'learning_rate': 0.01, 'subsample': 1.0,
          'colsample_bytree': 0.8, 'gamma': 5}

def Toplantı_hyper_parametredeneme_fonk(train_df, test_df, params={}):
    def dummies_cat_col(df):
        cat_cols = df.select_dtypes(include=["object", "category"]).columns
        df = pd.get_dummies(data=df, columns=cat_cols, drop_first=True)
        return df
    train_df = dummies_cat_col(train_df)
    test_df = dummies_cat_col(test_df)

    def scale_num_col(train, test):
        num_cols = list(train.select_dtypes(include=["float64", "int64"]).columns)
        num_cols.remove("loan_status")
        scaler = StandardScaler()
        scaler.fit(train[num_cols])
        train[num_cols] = scaler.transform(train[num_cols])
        test[num_cols] = scaler.transform(test[num_cols])
        return train, test
    train_df, test_df = scale_num_col(train_df, test_df)

    X = train_df.drop("loan_status", axis=1)  # Features
    y = train_df["loan_status"]  # Target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = XGBClassifier(**params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Değerlendirme metrikleri
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    #f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])

    print("\n🔍 Değerlendirme Metrikleri:")
    print(f"Accuracy     : {acc:.4f}")
    print(f"Precision    : {prec:.4f}")
    print(f"Recall       : {rec:.4f}")
    #print(f"F1-Score     : {f1:.4f}")
    print(f"ROC AUC      : {roc_auc:.4f}")

    return model,train_df,test_df
model, train_df, test_df = Toplantı_hyper_parametredeneme_fonk(train_df,test_df, params_1)

joblib.dump(best_model, "streamlit/model.pkl")