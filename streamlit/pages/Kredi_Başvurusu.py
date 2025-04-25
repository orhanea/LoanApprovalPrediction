import joblib
import plotly.graph_objects as go
import pandas as pd
import streamlit as st


# 🎨 Sayfa ayarları
st.set_page_config(
    page_title="Kredi Onay Sistemi",
    page_icon="💳",
    layout="centered"
)

# 🎯 Başlık ve tanıtım
st.markdown("<h1 style='text-align: center; color: #2E86C1;'>💳 Kredi Onay Sistemi</h1>", unsafe_allow_html=True)
st.markdown("""
<div style='text-align: center; color: #555; font-size: 16px;'>
    Bütçemiz Bütçeniz   
</div>
<hr>
""", unsafe_allow_html=True)

# 📋 Form ile kullanıcı girişi
with st.form("input_form"):
    col1, col2 = st.columns(2)

    with col1:
        person_age = st.number_input('👤 Yaş', min_value=18, max_value=100, value=30)
        person_income = st.number_input('💰 Yıllık Gelir (USD)', min_value=0, max_value=1000000, value=25000)
        person_home_ownership = st.selectbox('🏠 Ev Sahipliği', ['RENT', 'OWN', 'MORTGAGE', 'OTHER'])
        person_emp_length = st.number_input('🧑‍💼 Çalışma Süresi (yıl)', min_value=0, max_value=50, value=5)
        loan_intent = st.selectbox('🎯 Kredi Amacı', ['EDUCATION', 'MEDICAL', 'PERSONAL', 'VENTURE', 'DEBTCONSOLIDATION',
                                                     'HOMEIMPROVEMENT'])

    with col2:
        loan_grade = st.selectbox('📊 Kredi Puanı', ['A', 'B', 'C', 'D', 'E', 'F', 'G'])
        loan_amnt = st.number_input('💵 Kredi Tutarı (USD)', min_value=500, max_value=1000000, value=5000)
        loan_int_rate = st.number_input('📈 Faiz Oranı (%)', min_value=0.0, max_value=30.0, value=11.49)
        loan_percent_income = st.number_input('💸 Gelirin Ne Kadarı Krediye Ayrılacak? (%)', min_value=1.0, max_value=100.0, value=20.0)
        cb_person_default_on_file = st.selectbox('❗ Önceki Gecikme Var mı?', ['Y', 'N'])
        cb_person_cred_hist_length = st.number_input('📅 Kredi Geçmişi (yıl)', min_value=0, max_value=50, value=5)

    submitted = st.form_submit_button("📊 Tahmin Et")

# ⏳ Form gönderildiyse işle
if submitted:
    user_data = pd.DataFrame({
        'person_age': [person_age],
        'person_income': [person_income],
        'person_home_ownership': [person_home_ownership],
        'person_emp_length': [person_emp_length],
        'loan_intent': [loan_intent],
        'loan_grade': [loan_grade],
        'loan_amnt': [loan_amnt],
        'loan_int_rate': [loan_int_rate],
        'loan_percent_income': [loan_percent_income],
        'cb_person_default_on_file': [cb_person_default_on_file],
        'cb_person_cred_hist_length': [cb_person_cred_hist_length]
    })

    # Feature Engineering
    train_data = user_data.copy()
    train_data["NEW_income_interaction"] = train_data["person_income"] - train_data["loan_percent_income"]
    train_data['NEW_yas_borc_farki'] = train_data['loan_amnt'] - train_data['person_age'] * 100
    train_data['NEW_yıllık_ort_kredi_kullanım'] = train_data['loan_amnt'] / train_data['cb_person_cred_hist_length']

    # Model yükle
    current_dir = os.getcwd()
    model_path = os.path.join(current_dir,"streamlit","model_with_preprocessor.pkl")
    model = joblib.load(model_path)

    prediction = model.predict(train_data)

    if hasattr(model, "predict_proba"):
        probability = model.predict_proba(train_data)[0][1]
        probability_percent = round(probability * 100, 2)
    else:
        probability = None

    st.markdown("<hr>", unsafe_allow_html=True)

    if prediction[0] == 1:
        st.success("✅ Kredi Onaylandı!")
    else:
        st.error("❌ Kredi Reddedildi.")

    # Olasılık gösterimi
    if probability is not None:
        st.markdown(f"**📈 Onaylanma Olasılığı:** `%{probability_percent}`")

        # Gauge Chart (Plotly)
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=probability_percent,
            delta={'reference': 50, 'increasing': {'color': "green"}, 'decreasing': {'color': "red"}},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "royalblue"},
                'steps': [
                    {'range': [0, 50], 'color': 'salmon'},
                    {'range': [50, 100], 'color': 'lightgreen'}
                ],
                'threshold': {
                    'line': {'color': "black", 'width': 4},
                    'thickness': 0.75,
                    'value': 50
                }
            },
            title={'text': "Kredi Onay Skoru (%)"}
        ))
        st.plotly_chart(fig)

        if probability_percent < 50:
            st.warning("Onay ihtimali düşük. Gelir/kredi oranınızı gözden geçirmeniz önerilir.")

    # Kullanıcı verilerini göster (isteğe bağlı)
    with st.expander("📄 Müşteri Verilerimiz"):
        st.dataframe(user_data)

# Footer
st.markdown("""
<hr>
<div style='text-align: center; color: gray; font-size: 14px;'>
    Geliştirici: DatAkışı Ekibi | Gururla Sunar | 📅 2025
</div>
""", unsafe_allow_html=True)


