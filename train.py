import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import nltk
import pickle
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('punkt_tab',quiet=False)
nltk.download('vader_lexicon')
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
vectorizer = CountVectorizer(max_features=5000)
def lemmatizerfun(x):
  removedstoppedword=[]
  for review in x:
      words = word_tokenize(review.lower())
      cleaned_review = []
      for word in words:
          if word.isalpha() and word not in stop_words:
              lemmatizer_word=lemmatizer.lemmatize(word)
              cleaned_review.append(lemmatizer_word)
      removedstoppedword.append(" ".join(cleaned_review))

  return removedstoppedword
def vectorize_reviews(reviews):
    return vectorizer.transform(reviews).toarray()
dataset = pd.read_csv('fake_reviews_dataset1.csv')
dataset = dataset.dropna()
dataset['label'] = dataset['label'].map({"CG": 0, "OR": 1})
cleaned_reviews = lemmatizerfun(dataset['text_'])
train_x_vector = vectorizer.fit_transform(cleaned_reviews).toarray()
train_y = dataset['label']
x_train, x_test, y_train, y_test = train_test_split(
    train_x_vector, train_y, test_size=0.2, random_state=42
)
model = LogisticRegression()
model.fit(x_train, y_train)
with open("model.pkl", "wb") as file:  
    pickle.dump(model, file) 
with open("vectorizer.pkl", "wb") as file:  
    pickle.dump(vectorizer, file) 


