#!/usr/bin/env python
# coding: utf-8

# In[1]:


import nltk


# In[18]:


nltk.download('punkt_tab')
nltk.download('punkt')
nltk.download('wordnet')


# In[28]:


import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB


# In[13]:


#tokenisation

text = 'Natural language processing (NLP) is a field of computer science and a subfield of artificial intelligence that aims to make computers understand human language. NLP uses computational linguistics, which is the study of how language works, and various models based on statistics, machine learning, and deep learning. '
tokens = word_tokenize(text)
print(tokens)


# In[14]:


#Stemming
stemmer = PorterStemmer()
stemmed_tokens = [stemmer.stem(token) for token in tokens]
print(stemmed_tokens)


# In[19]:


#lemmatization
lemmatizer = WordNetLemmatizer()
lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
print(lemmatized_tokens)


# In[23]:


#text Classification

documents=[("This is Positive review",'positive'),("This is negative review",'negative'),("This is neutral review",'neutral')]


# In[32]:


vectorizer = CountVectorizer()
x = vectorizer.fit_transform ([doc for doc,label in documents])
y = [label for doc,label in documents]

clf = MultinomialNB()
clf.fit(x,y)


# In[33]:


new_text = "this is very bad"
new_x = vectorizer.transform([new_text])
predicted_sentiment = clf.predict(new_x)
print(predicted_sentiment)


# In[ ]:




