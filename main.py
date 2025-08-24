import numpy as np
import pandas as pd 
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import accuracy_score

import nltk
nltk.download('stopwords')

news_dataset = pd.read_csv('train.csv')
print(news_dataset.shape)

# Counting number of missing numbers
# print(news_dataset.isnull().sum())

# Replacing missing numbers as null 
news_dataset = news_dataset.fillna('')

# Merging author name and news title
news_dataset['content'] = news_dataset['author'] + ' ' + news_dataset['title']

# Separarting data and labels
X = news_dataset.drop(columns='label', axis=1)
Y = news_dataset['label']

# Stemming
port_stem = PorterStemmer()
def stemming(content):
    stemmed_content = re.sub('[^a-zA-Z]',' ', content)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [port_stem.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
    stemmed_content = ' '.join(stemmed_content)
    return stemmed_content

news_dataset['content'] = news_dataset['content'].apply(stemming)

# Preparing data and label
X = news_dataset['content'].values
Y = news_dataset['label'].values

# Converting text data to numerical data
vectorizer = TfidfVectorizer()
vectorizer.fit(X)

X = vectorizer.transform(X)

# Splitting dataset into training and testing
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, stratify=Y,random_state=2)

# Training
model = LogisticRegression()
model.fit(X_train,Y_train)

# Accuracy score on training data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
print(f'Accuracy score on training data: {training_data_accuracy}')

# Accuracy score on test data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print(f'Accuracy score on test data: {test_data_accuracy}')

# Making a predictive system
# X_new_1 = X_test[0]
# print(X_new_1)
# X_new_2 = X_test[1]

# prediction = model.predict(X_new_1)
# prediction = model.predict(X_new_2)
# print(prediction)

'''if (prediction[0]==0):
    print('Fake news')
else:
    print('Real news') '''

# print(Y_test[0])
# print(Y_test[2])

# Function to preprocess custom input
def preprocess_input(text):
    stemmed_content = re.sub('[^a-zA-Z]', ' ', text)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [port_stem.stem(word) for word in stemmed_content if word not in stopwords.words('english')]
    stemmed_content = ' '.join(stemmed_content)
    return stemmed_content

# Example custom input
custom_news = "Breaking: Government confirms new policies to support startups and innovation."
print(custom_news)

# Preprocess the input
preprocessed_news = preprocess_input(custom_news)

# Vectorize the input
vectorized_input = vectorizer.transform([preprocessed_news])

# Predict
prediction = model.predict(vectorized_input)

# Output result
if prediction[0] == 0:
    print("Prediction: Fake news")
else:
    print("Prediction: Real news")
