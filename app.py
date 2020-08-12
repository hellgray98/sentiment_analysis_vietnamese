from flask import Flask, request, render_template
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import twitter_samples, stopwords
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
from nltk import FreqDist, classify, NaiveBayesClassifier
import nltk
import re, string, random
import numpy as np 
import pandas as pd 
import re
import nltk 
import matplotlib.pyplot as plt
from string import punctuation
import pickle
    
app = Flask(__name__)

@app.route('/')
def my_form():
    return render_template('form.html')

@app.route('/', methods=['POST'])
def my_form_post():
    data_source_url = 'comment1.csv'
    df = pd.read_csv(data_source_url)
    df.head()

    plot_size = plt.rcParams["figure.figsize"] 

    plot_size[0] = 8
    plot_size[1] = 6
    plt.rcParams["figure.figsize"] = plot_size 

    comments = df.iloc[:, 0].values
    labels = df.iloc[:,4].values

    print(comments)
    print(labels)
    stop_word = []
    f= open("stop_word.txt",encoding="utf-8")
    text = f.read()

    for word in text.split() :
        stop_word.append(word)
    f.close()

    punc = list(punctuation)
    stop_word = stop_word + punc

    processed_comments = []

    for sentence in range(0, len(comments)):
        processed_comment = re.sub(r'\W', ' ', str(comments[sentence]))
        processed_comment= re.sub(r'\s+[a-zA-Z]\s+', ' ', processed_comment)
        processed_comment = re.sub(r'\^[a-zA-Z]\s+', ' ', processed_comment) 
        processed_comment = re.sub(r'\s+', ' ', processed_comment, flags=re.I)
        processed_comment = re.sub(r'^b\s+', '', processed_comment)
        processed_comment = processed_comment.lower()
        processed_comments.append(processed_comment)

    from sklearn.feature_extraction.text import TfidfVectorizer
    vectorizer = TfidfVectorizer (max_features=3900, min_df=5, max_df=0.8, stop_words=stop_word)

    vectorizer.fit(processed_comments)
    X = vectorizer.transform(processed_comments).toarray()

    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.3, random_state=0)

    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import classification_report
    model = LogisticRegression()
    model.fit(X_train,y_train)
    y_pre = model.predict(X_test)
    print(classification_report(y_test,y_pre))

    filename = 'finalized_model.sav'
    pickle.dump(model, open(filename, 'wb'))

    loaded_model = pickle.load(open(filename, 'rb'))
    result = loaded_model.score(X_test, y_test)

    h1 =[["bọn họ là người xấu"]]
    text1 = request.form['text1'].lower()
    sentences = []
    sentences.append(text1)
    sentence_vector = vectorizer.transform([sentences[0]])
    output = model.predict(sentence_vector)
    str(output)
    a = "Negative"
    b = "Positive"
    cincin = " "
    if (output == ['positive']):
        cincin = b
    else:
         cincin = a
    # for i in text1:
    #     kt = vectorizer.fit_transform(i)
    #     print(i)
    #     text2=loaded_model.predict(kt)
    #     print(loaded_model.predict(kt))    
    #-----------------------------------------------------------
    return render_template('form.html',final = output, text1=text1, text2 = cincin)
if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5002, threaded=True)
