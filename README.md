<h1>MachineLearning_Step6_NLP_OCR</h1>

# NLP

Natural Language Processing (NLP), bilgisayarların insan dili anlamasını ve işlemesini sağlamak için kullanılan bir yapay zeka dalıdır. NLP, metin verilerini analiz etmek, anlamak ve çıkarımlar yapmak için kullanılır.

NLP'nin ana uygulama alanlarından bazıları şunlardır:

- **Dil Çevirisi:** Farklı diller arasında otomatik çeviri yapmak için NLP kullanılır.
- **Sentiment Analizi:** Bir metnin duygusal tonunu (pozitif, negatif, nötr) belirlemek için NLP kullanılır.
- **Chatbots ve Sesli Asistanlar:** Kullanıcıların doğal dildeki sorularını anlamak ve yanıtlamak için NLP kullanılır.
- **Metin Özeti:** Büyük metinlerin anahtar noktalarını özetlemek için NLP kullanılır.
- **İsimli Varlık Tanıma:** Metinden isimler, yerler, tarihler gibi özel bilgileri çıkarmak için NLP kullanılır.

NLP, dilin karmaşıklığı ve belirsizliği nedeniyle bir dizi zorluk sunar. Örneğin, aynı kelimenin farklı bağlamlarda farklı anlamlara gelebilmesi veya dilin ironi ve argo gibi nüansları anlamak zordur. Bu zorlukların üstesinden gelmek için, NLP genellikle makine öğrenmesi ve derin öğrenme tekniklerini kullanır.

Python'da Natural Language Processing (NLP) için genellikle NLTK (Natural Language Toolkit) veya Spacy gibi kütüphaneler kullanılır. Aşağıda, bir metinden duraklama (stop) kelimelerini çıkarmak için NLTK kullanılan bir örnek verilmiştir.

```python
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# NLTK'nin duraklama kelimelerini indir
nltk.download('punkt')
nltk.download('stopwords')

# Analiz edilecek metin
text = "This is a sample sentence, showing off the stop words filtration."

# Metni kelimelere ayır
word_tokens = word_tokenize(text)

# Duraklama kelimelerini filtrele
filtered_sentence = [word for word in word_tokens if not word in stopwords.words()]

print("Original Sentence: ", word_tokens)
print("Filtered Sentence: ", filtered_sentence)
```

Bu kod, bir metni kelimelere ayırır ve İngilizce'deki duraklama kelimelerini (örneğin "is", "a", "the") filtreler. Duraklama kelimeleri genellikle NLP'de çıkarılır, çünkü genellikle metnin anlamına çok az katkıda bulunurlar.

## *****

Natural Language Processing (NLP) is a branch of artificial intelligence that focuses on the interaction between computers and humans through natural language. The ultimate objective of NLP is to read, decipher, understand, and make sense of human language in a valuable way.

NLP involves several key tasks and applications, including:

- **Machine Translation:** Translating text or speech from one language to another.
- **Speech Recognition:** Converting spoken language into written form.
- **Sentiment Analysis:** Determining the emotional tone behind words to gain an understanding of the attitudes, opinions, and emotions expressed within an online mention.
- **Information Extraction:** Identifying and extracting structured information from unstructured document collections.
- **Chatbots and Virtual Assistants:** Using NLP to interact with users in a natural, human-like way.

NLP is a complex field because human language is rarely precise or straightforward. It often involves the use of machine learning and deep learning techniques to understand the nuances and semantics of human language. Libraries like NLTK, SpaCy, and transformers in Python are commonly used for NLP tasks.

In Python, libraries like NLTK (Natural Language Toolkit), SpaCy, and transformers are commonly used for Natural Language Processing (NLP) tasks. Here's an example of using SpaCy for named entity recognition, a task where you identify important named entities in the text such as people, places, dates, etc.

```python
import spacy

# Load the English language model
nlp = spacy.load('en_core_web_sm')

# The text we want to examine
text = "Apple is looking at buying U.K. startup for $1 billion"

# Process the text
doc = nlp(text)

# Print the named entities found
for ent in doc.ents:
    print(ent.text, ent.label_)
```

In this code, we first load the English language model. We then create a `Doc` object by applying the model to our text. The `Doc` object that's returned is then traversed to print out the named entities that were found. For each named entity, we print the text of the entity and its label.



# OCR

Optical Character Recognition (OCR), basılı veya yazılı metni dijital metne dönüştürme işlemidir. OCR, bir görüntüyü (genellikle taramış olduğunuz bir belge veya bir fotoğraf) alır ve içindeki metni tanımlar. Bu, metni düzenlenebilir, aranabilir ve dijital olarak işlenebilir hale getirir.

OCR'nin ana kullanım alanlarından bazıları şunlardır:

- **Belge Dijitalleştirme:** Fiziksel belgeleri dijital formata dönüştürmek için OCR kullanılır. Bu, belgelerin aranabilir ve düzenlenebilir hale gelmesini sağlar.
- **Veri Girişi Otomasyonu:** OCR, veri girişi işlemlerini otomatikleştirmek için kullanılır, örneğin faturaların, çeklerin veya posta kodlarının okunması.
- **Lisans Plakası Tanıma:** OCR, araç lisans plakalarını otomatik olarak okumak için kullanılır, örneğin hız kameraları veya otopark girişlerinde.
- **El Yazısı Tanıma:** OCR, el yazısı notları veya belgeleri dijital metne dönüştürmek için kullanılır.

OCR, metni tanımlamak için genellikle makine öğrenmesi ve görüntü işleme tekniklerini kullanır. Ancak, OCR'nin doğruluğu, görüntünün kalitesi, metnin düzeni ve yazı tipi gibi faktörlere bağlıdır.

Python'da OCR (Optical Character Recognition) işlemleri için genellikle pytesseract kütüphanesi kullanılır. Bu kütüphane, açık kaynaklı bir OCR aracı olan Tesseract'ı Python'da kullanılabilir hale getirir.

Aşağıda, bir görüntüden metin çıkarmak için pytesseract kullanılan bir örnek verilmiştir.

```python
from PIL import Image
import pytesseract

# Tesseract executable path
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Windows
# pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'  # Linux

# Open image
img = Image.open('image.png')

# Use Tesseract to do OCR on the image
text = pytesseract.image_to_string(img)

print(text)
```

Bu kod, bir görüntüyü açar ve Tesseract'ı kullanarak görüntüden metin çıkarır. Tesseract'ın yolu, işletim sistemine bağlı olarak ayarlanmalıdır. Bu kod, İngilizce metni tanımlar. Farklı bir dilde metin tanımlamak için, ilgili dil verilerini indirmeniz ve Tesseract'a dil parametresi olarak vermeniz gerekir.

## *****

Optical Character Recognition (OCR) is a technology used to convert different types of documents, such as scanned paper documents, PDF files or images captured by a digital camera, into editable and searchable data.

OCR works by analyzing the shapes and patterns of the characters in the scanned image or document, then converting them into a form that a computer can manipulate (such as a text file or digital document).

Key applications of OCR include:

- **Data Entry Automation:** OCR can be used to automate data entry processes, such as entering data from paper documents into a database.
- **Document Digitization:** OCR is used to digitize printed documents so they can be edited, searched, and stored more compactly.
- **License Plate Recognition:** OCR technology is often used in traffic monitoring systems to identify vehicles by their license plates.
- **Handwriting Recognition:** Advanced OCR systems can recognize handwritten text, and are used in fields like banking to process cheques and in mail sorting centers to route mail.

While OCR is a powerful tool, it's not perfect. The accuracy of OCR can be affected by factors such as the quality of the original document, the font used, and the system's ability to recognize different characters and layouts.

In Python, the pytesseract library is commonly used for Optical Character Recognition (OCR) tasks. This library is a wrapper for Google's Tesseract-OCR Engine. Here's an example of using pytesseract to extract text from an image:

```python
from PIL import Image
import pytesseract

# Specify the path to the tesseract OCR engine
# For Windows users, this is typically: r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'
# For Linux users, this is typically: '/usr/bin/tesseract'
pytesseract.pytesseract.tesseract_cmd = r'Your_Path_to_tesseract.exe'

# Open an image file
img = Image.open('test_image.png')

# Use pytesseract to convert the image into text
text = pytesseract.image_to_string(img)

# Print the text
print(text)

```

In this code, we first specify the path to the Tesseract OCR engine. We then open an image file using the PIL library. We use pytesseract to convert the image into text, and then print the text.

Please replace `'test_image.png'` with the path to your image file, and `'Your_Path_to_tesseract.exe'` with the path to your Tesseract OCR engine.
