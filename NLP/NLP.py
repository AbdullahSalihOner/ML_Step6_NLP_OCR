#NLP uygulması / NLP application
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt  

data = pd.read_csv("NLPlabeledData.tsv", delimiter = "\t", quoting = 3) #quoting = 3 -> tırnak işaretlerini yok say / delimiter = "\t" -> tsv dosyası olduğu için tab ile ayrılmıştır
#%%
data = pd.read_table("NLPlabeledData.tsv") #delimiter = "\t" -> tsv dosyası olduğu için tab ile ayrılmıştır / read_table -> read_csv ile aynı işlevi görür
#%%
#veri ön işleme
from nltk.corpus import stopwords #stopwords -> gereksiz kelimeler / useless words
nltk.download("stopwords") #stopwords indir / download stopwords
liste = stopwords.words("english") #ingilizce stopwords listesi / english stopwords list
liste1 = stopwords.words("turkish") #turkçe stopwords listesi / turkish stopwords list
#%%
#veri temizleme işlemleri 1 / data cleaning operations 1
ornek_metin = data.review[0] # 0. indeksteki yorumu al / get the review at index 0

#%%
#veri temizleme işlemleri 2 / data cleaning operations 2
from bs4 import BeautifulSoup #html taglarını temizlemek için kullanılır / used to clean html tags
ornek_metin1 = BeautifulSoup(ornek_metin,features="lxml").get_text() #html taglarını temizle / clean html tags
#%%
#veri temizleme işlemleri 3 / data cleaning operations 3
import re #regular expression -> düzenli ifadeler için kullanılır  / used for regular expressions
ornek_metin2 = re.sub("[^a-zA-Z]"," ",ornek_metin1) #sadece a-z ve A-Z harfleri kalacak şekilde temizle  / clean only a-z and A-Z letters

#%%
#veri temizleme işlemleri 4 / data cleaning operations 4
ornek_metin3 = ornek_metin2.lower() #bütün harfleri küçült   / lower all letters

#%%
#veri temizleme işlemleri 5 / data cleaning operations 5
ornek_metin4 = ornek_metin3.split() #kelimeleri ayır / split the words

#%%
#veri temizleme işlemleri 6 / data cleaning operations 6
swords = set(stopwords.words("english")) #ingilizce stopwords listesi / english stopwords list
prnek_metin5 = [word for word in ornek_metin4 if not word in swords] #stopwordsleri temizle / clean stopwords
#%%

#işlem fonksiyonu / process function
def islem(review):
    review = BeautifulSoup(review,features="lxml").get_text() #html taglarını temizle  / clean html tags
    review = re.sub("[^a-zA-Z]"," ",review) #sadece a-z ve A-Z harfleri kalacak şekilde temizle  / clean only a-z and A-Z letters
    review = review.lower() #bütün harfleri küçült  / lower all letters
    review = review.split() #kelimeleri ayır    / split the words
    swords = set(stopwords.words("english")) #ingilizce stopwords listesi    / english stopwords list
    review = [word for word in review if not word in swords] #stopwordsleri temizle / clean stopwords
    return (" ".join(review)) #kelimeleri birleştir     / join the words

#%%
# 1000 yorumu temizle ve ekrana yaz / clean 1000 reviews and print to the screen
train_x_tum = []
for i in range(len(data["review"])): #bütün yorumları temizle / clean all reviews
    if(i+1)%1000 == 0: #her 1000 yorumda bir ekrana yaz / print to the screen every 1000 reviews
        print(str(i+1) + " yorum temizlendi")
    train_x_tum.append(islem(data["review"][i])) #temizlenmiş yorumları listeye ekle / add cleaned reviews to the list
    
#%%
# veriyi train ve test olarak ayırma / split the data into train and test
from sklearn.model_selection import train_test_split #veriyi train ve test olarak ayırma   / split the data into train and test  
x = train_x_tum #x -> yorumlar / x -> reviews
y = np.array(data["sentiment"]) #y -> yorumların duyguları  0 -> negatif 1 -> pozitif  / y -> sentiments of the reviews  0 -> negative 1 -> positive
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2,random_state = 42) #veriyi train ve test olarak ayırma   / split the data into train and test

#%%
from sklearn.feature_extraction.text import CountVectorizer #kelimeleri vektöre çevirme  / convert the words to vector
cv = CountVectorizer(max_features = 20000) #en çok kullanılan 20000 kelimeyi al  / take the most used 20000 words
xtrain = cv.fit_transform(x_train) #x_train'i vektöre çevir / convert x_train to vector
xtrain = xtrain.toarray() #xtrain'i array'e çevir / convert xtrain to array

xtest = cv.transform(x_test) #x_test'i vektöre çevir / convert x_test to vector
xtest = xtest.toarray() #xtest'i array'e çevir / convert xtest to array

#%%
"""

#model oluşturma
from sklearn.naive_bayes import GaussianNB #naive bayes algoritması
gnb = GaussianNB() #model oluştur
gnb.fit(xtrain,y_train) #modeli eğit    

"""
#%%
"""
#modeli test etme
from sklearn.metrics import roc_auc_score #roc eğrisi
y_pred = gnb.predict(xtest) #modeli test et 
accuracy = roc_auc_score(y_test,y_pred) #doğruluk oranı
print("Accuracy: %.5f%%" % (accuracy * 100.0)) #doğruluk oranı 
 
"""
#%%
#model oluşturma / model creation
from sklearn.ensemble import RandomForestClassifier #random forest algoritması   / random forest algorithm
rfc = RandomForestClassifier(n_estimators = 100) #model oluştur / create model
rfc.fit(xtrain,y_train) #modeli eğit / train the model
xtest = cv.transform(x_test) #x_test'i vektöre çevir / convert x_test to vector

xtest = xtest.toarray() #xtest'i array'e çevir / convert xtest to array
print(xtest.shape)

#%%
#modeli test etme / test the model
from sklearn.metrics import roc_auc_score #roc eğrisi / roc curve
y_pred = rfc.predict(xtest) #modeli test et     / test the model
accuracy = roc_auc_score(y_test,y_pred) #doğruluk oranı / accuracy
print("Accuracy: %.5f%%" % (accuracy * 100.0)) #doğruluk oranı / accuracy

#%%

manuel_cümle = "The movie was a huge disappointment with its predictable plot, poor acting, and dull cinematography; it completely failed to capture the essence of the original book."

# Ön işleme fonksiyonunu kullanarak cümleyi işle ve vektörleştirme  / Process the sentence using the preprocessing function and vectorize it
manuel_cümle_islenmis = islem(manuel_cümle)

# CountVectorizer ile vektörleştirme / Vectorize using CountVectorizer
manuel_cümle_vektör = cv.transform([manuel_cümle_islenmis])

# Rondom Forest ile tahmin etme / Predict using Random Forest
tahmin = rfc.predict(manuel_cümle_vektör)

# Tahmini yazdırma   / Print the prediction
if tahmin[0] == 0:
    print("Negatif")
else:
    print("Pozitif")
    
#%%
from sklearn.metrics import accuracy_score  

accuracy = accuracy_score(y_test,y_pred) #doğruluk oranı / accuracy
print("Accuracy: %.5f%%" % (accuracy * 100.0)) #doğruluk oranı / accuracy

#%%

# Function to predict sentiment of a given sentence using the trained model and CountVectorizer / Eğitilmiş model ve CountVectorizer kullanarak verilen cümlenin duygusunu tahmin etmek için kullanılan fonksiyon
def predict_sentiment(sentence):
    # Clean and preprocess the sentence using the preprocessing function / Ön işleme fonksiyonunu kullanarak cümleyi temizle ve işle
    sentence_processed = islem(sentence)
    # Transform the sentence to the same format as the training data / Cümleyi eğitim verisiyle aynı formata dönüştür
    sentence_vectorized = cv.transform([sentence_processed]).toarray()
    # Predict using the trained RandomForestClassifier model / Eğitilmiş RandomForestClassifier modelini kullanarak tahmin et
    prediction = rfc.predict(sentence_vectorized) 
    # Return the prediction as "Positive" if the predicted value is 1, otherwise "Negative" / Tahmin edilen değer 1 ise "Positive" olarak döndür, aksi takdirde "Negative" olarak döndür
    return "Positive" if prediction[0] == 1 else "Negative"

# Asking for user input / Kullanıcı girişi isteme
user_input_sentence = input("Enter a sentence to determine its sentiment (Positive/Negative): ")

# Predicting the sentiment of the user input sentence / Kullanıcı giriş cümlesinin duygusunu tahmin etme
predicted_sentiment = predict_sentiment(user_input_sentence)
print(f"The sentiment of the entered sentence is: {predicted_sentiment}")
