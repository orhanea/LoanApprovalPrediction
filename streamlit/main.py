import streamlit as st
import requests
from io import BytesIO

st.set_page_config(
    page_title="Main",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Başlık
st.title("DATABANK")

st.divider()

st.title("Biz Kimiz?")

# 2 sütun oluşturma (2:1 oranında)
col1, col2 = st.columns([2, 1])

# Birinci sütuna açıklama metni ekleme
with col1:
    st.write("""
        **Data Bank** olarak, finansal hizmetler sektöründe yenilikçi çözümler sunan bir dijital banka olarak, 
        müşteri odaklı bir yaklaşım benimsemekteyiz. Amacımız, müşterilerimizin ihtiyaçlarına en iyi şekilde 
        hizmet vermek ve finansal dünyada daha erişilebilir ve şeffaf bir deneyim sağlamaktır.

        **Misyonumuz**:
        Modern teknolojiler kullanarak, geleneksel bankacılık hizmetlerini dijital platformlara taşımak ve 
        her birey için bankacılık işlemlerini daha hızlı, güvenli ve kullanımı kolay hale getirmektir. 
        Müşterilerimize, finansal yönetimlerini daha etkili ve verimli bir şekilde gerçekleştirebilecekleri 
        araçlar sunuyoruz.

        **Vizyonumuz**:
        Data Bank, gelecekteki bankacılığın öncüsü olmayı hedeflemektedir. Finansal hizmetlerdeki dijital dönüşümü 
        hızlandırarak, tüm dünyada müşteri memnuniyeti ve güveni odaklı bir banka olmayı amaçlıyoruz. 
        Teknolojik altyapımızı sürekli geliştirerek, müşterilerimize her zaman en iyi hizmeti sunmayı hedefliyoruz.
    """)

# İkinci sütuna bir resim ya da başka içerik ekleme (örneğin bir logo)
with col2:
    # Google Drive dosya ID'si
    file_id = "1Hyhxaupp63zdBCWADgf2UIoG0ferW_b-"  # Burada dosyanın ID'si
    # Google Drive dosyasını indirme linki oluşturma
    file_url = f"https://drive.google.com/uc?id={file_id}"
    # `requests` ile dosyayı indirme
    response = requests.get(file_url)
    # Eğer istek başarılı olduysa (status code 200)
    if response.status_code == 200:
        # Resmi BytesIO ile al
        image_data = BytesIO(response.content)

        # Resmi Streamlit ile gösterme
        st.image(image_data,width=450)
    else:
        st.error("Resim yüklenirken bir hata oluştu.")

    # Veya burada kısa bir bilgi de gösterebilirsiniz
    st.write("Data Bank hakkında daha fazla bilgi edinmek için bizimle iletişime geçin.")






