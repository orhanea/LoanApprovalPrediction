# LoanApprovalPrediction

Projenin asıl amacı kullanıcıların kredisinin onaylanıp onaylanmayacağını tahmin etmektir. Kullanacağımız kaggle yarışmasının veri seti (hem eğitim hem de test), Kredi Onayı Tahmini veri seti üzerinde eğitilen derin öğrenme modelinden oluşturuldu. Eğitim setinde hedef değişken mevcut ama test setinde hedef değişken mevcut değildir. Bu mevcut olmayan değişkeni biz daha sonradan kendimiz bulduğumuz orijinal veri setini de kullanarak oluşturacağımız model ile tahminleme yapacağız. Veri setlerini aşağıda bulabilirsiniz. Daha sonrasında değişkenlerin gerekli açıklamalarını ayrıntılı grafikler ile bu dokümanda paylaşacağız.

Kaggle Playground Series : <https://www.kaggle.com/competitions/playground-series-s4e10/data>

Orijinal veri seti : <https://www.kaggle.com/datasets/laotse/credit-risk-dataset>

## 🚩 Değişkenlerimiz

1. [person_age](#person_age)
2. [person_income](#person_income)
3. [person_home_ownership](#person_home_ownership)
4. [person_emp_length](#person_emp_length)
5. [loan_intent](#loan_intent)
6. [loan_grade](#loan_grade)
7. [loan_amnt](#loan_amnt)
8. [loan_int_rate](#loan_int_rate)
9. [loan_percent_income](#loan_percent_income)
10. [cb_person_default_on_file](#cb_person_default_on_file)
11. [cb_person_cred_hist_length](#cb_person_cred_hist_length)
12. [loan_status](#loan_status)

## person_age: Kişinin yaşı

- Değişken türü hem train hem test için int64 .
- 40 yaş altı yoğunluklu bir müşteri portfolyomuz var. 
- 40 yaş sonrası müşterilerimiz çok seyrek.
- Train ve Test verilerimiz birbirine benzer dağılım göstermektedir.

  ![Alt text](images/person_age.png)

## person_income: Kişinin yıllık geliri (USD cinsinden).

- Değişken türü hem train hem test için int64
- 205.000 USD altında yıllık geliri olan müşterilerimiz verimizin çoğunluğu oluşturuyor fakat daha üst yıllık gelir elde eden müşterilerimizde var ama çok az bir azınlık
- Train ve test veri setlerimiz aynı dağılıma sahip

  ![Alt text](images/person_income.png)

## person_home_ownership: Kişinin ev sahipliği durumu

- Değişken türü hem train hem test için object

   -  RENT: Kirada
   - OWN: Ev sahibi
   - MORTGAGE: Kredili ev sahibi
   - OTHER: Diğer

- Train ve test veri setlerimiz aynı dağılıma sahip

  ![Alt text](images/person_home_ownership.png)

  Original 
  
  ![Alt text](images/person_home_ownership_original.png)

## person_emp_length: Başvuranın kaç yıldır çalıştığı

- Değişken türü hem train hem test için float64

  ![Alt text](images/person_emp_length.png)

## loan_intent: Başvuranın krediye ihtiyaç duymasının nedeni.

- Değişken türü hem train hem test için object

  - **EDUCATION** – **Eğitim**  
    Eğitimle ilgili harcamaları finanse etmek için kullanılan kredileri ifade eder (örneğin, okul ücreti, kitap, kurs).
  - **MEDICAL** – **Tıbbi / Sağlık**  
    Sağlıkla ilgili giderleri karşılamak için alınan kredileri ya da sağlık hizmetlerini ifade eder (örneğin, ameliyat, tedavi, ilaç).
  - **PERSONAL** – **Kişisel**  
    Genel bireysel ihtiyaçlar için alınan kredilerdir. Özel bir amaca bağlı değildir; tatil, alışveriş veya borç kapatma gibi çok çeşitli kullanımları olabilir.
  - **VENTURE** – **Girişim / Yatırım**  
    Genellikle yeni iş kurma ya da mevcut işini büyütme amacıyla yapılan yatırımları veya girişimleri ifade eder.
  - **DEBT CONSOLIDATION** – **Borç Birleştirme**  
    Birden fazla borcun tek bir kredi altında toplanarak daha kolay ödenmesini sağlayan finansal yöntemdir. Faiz oranı düşürülebilir ve ödeme takvimi sadeleştirilebilir.
  - **HOME IMPROVEMENT** – **Ev Yenileme / İyileştirme**  
    Ev tadilatı, onarımı ya da geliştirmesi için kullanılan kredilerdir (örneğin, mutfak yenileme, çatı tamiri, enerji verimliliği artırma).
        
    ![Alt text](images/loan_intent.png)
        
## loan_grade: Başvuranın kredileri geri ödemede ne kadar güvenilir olduğunu gösteren bir puan.

- Değişken türü hem train hem test için object

- A,B,C,D,E,F,G( A en güvenilir olan segment diğerleri gittikçe azalıyor)

  ![Alt text](images/loan_grade.png)
  
## loan_amnt: Başvuranın borç almak istediği para miktarı.

- Değişken türü hem train hem test için int64

![Alt text](images/loan_amount.png)

## loan_int_rate: Krediye uygulanan faiz oranı.

- Değişken türü hem train hem test için float64

- (kredi faiz oranı) ifadesi, genellikle bir kredinin yıllık faiz oranını yüzde (%) cinsinden gösterir. 11.49 değeri, bu kredi için yıllık %11.49 faiz uygulandığını gösteriyor.

![Alt text](images/loan_int_grade.png)

## loan_percent_income: Başvuranın gelirinin ne kadarlık kısmının kredi ödemelerine gideceği.

- Hem train hem test için türü float64

![Alt text](images/loan_percent_income.png)

## cb_person_default_on_file: Başvuranın daha önce bir krediyi geri ödemede başarısız olup olmadığını gösterir.

- Hem train hem test için türü object

![Alt text](images/cb_person_default_on_file.png)

## cb_person_cred_hist_length: Başvuranın kredi geçmişinin ne kadar uzun olduğu

- Hem train hem test için türü int64

![Alt text](images/cb_person_cred_hist_length.png)

## loan_status: Kredinin onaylandığını veya reddedildiğini gösterir.
