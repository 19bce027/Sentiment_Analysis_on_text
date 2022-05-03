import pickle
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
import shutup
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from flask import Flask,render_template, redirect, request
import os
import re
from textblob import TextBlob
from chat_word_remove import chat_words_conversion


shutup.please()

# os.system('python -m nltk.downloader all')

app = Flask(__name__)

VOCAB_SIZE = 203243
MAXLEN = 2441

model = load_model('LSTM_rnn.h1')

stop_words = set(stopwords.words('english'))
with open(f'cv.pkl', 'rb') as f:
    countv = pickle.load(f)
with open(f'tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)
with open(f'Random_forest_clf.pkl', 'rb') as f:
    random_for = pickle.load(f)

def correct(text):
    textblb = TextBlob(text) 
    return textblb.correct().string

def remove_special_characters(text, remove_digits=True):
    pattern = r'[^a-zA-Z0-9\s]' if not remove_digits else r'[^a-zA-Z\s]'
    text = re.sub(pattern, '', text)
    return text

def tokenized(text) :
    tt = TweetTokenizer()
    text = str(text).lower()
    text = tt.tokenize(text)
    return text

def stopword_removal(text) :
    text  = list([word for word in text if word not in (stop_words)])
    return text

def stemming(text) :
    lem = WordNetLemmatizer()
    text = list([lem.lemmatize(word) for word in text])
    return text

def finaltext(text) : 
    text = ' '.join([str(elem) for elem in text])
    return text

def countvector(text) :
    count_vector = countv.transform(text)
    return count_vector

def tokenizer_rnn(text) :
    text = tokenizer.texts_to_sequences(text)
    return text

def padding(text) :
    text = pad_sequences(text, MAXLEN+1,padding='pre')
    return text

def predict_ml(text) :
    text = correct(text)
    text = chat_words_conversion(text)
    text = remove_special_characters(text)
    tokenword = tokenized(text)
    # sword = stopword_removal(tokenword)
    stemmed = stemming(tokenword)
    final = finaltext(stemmed)
    final = final.split('\n')
    count_vector = countvector(final)
    print(final)
    return random_for.predict(count_vector)

def predict_lstm(text) :
    text = correct(text)
    text = chat_words_conversion(text)
    text = remove_special_characters(text)
    tokenword = tokenized(text)
    # sword = stopword_removal(tokenword)
    stemmed = stemming(tokenword)
    final = finaltext(stemmed)   
    final = final.split('\n')
    print(final)
    token_rnn = tokenizer_rnn(final)
    after_pad = padding(token_rnn)
    pred = model.predict(after_pad,verbose=False)
    return pred

@app.route('/sentiment', methods = ['GET','POST'])
def sentiment():
    review = request.form.get('review')
    if review == "" :
        error = "Please Enter any one value"
        return render_template('index.html', error=error)
    else :
        ml_per = predict_ml(review)[0]
        print(type(ml_per))
        if ml_per==0 :
            ml_sen = 'negative'
        else :
            ml_sen = 'positive'
        dl_per = predict_lstm(review)[0][0]
        print(dl_per)
        if dl_per>=0.50:
            dl_sen = 'positive'
        else :
            dl_sen = 'negative'
        return render_template('sentiment.html', name=review,ml_sen=ml_sen,dl_sen=dl_sen,dl_per=round(float(dl_per*100),2))

# predict_lstm("The movie was not good.")
# predict_ml("The movie was not good.")

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/check')
def start():
    return "Listening"

app.run(debug=True,port=3000)
# nltk.download('stopwords')
# nltk.download('wordnet')