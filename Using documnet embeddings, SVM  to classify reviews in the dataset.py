#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Loading Yelp Dataset - reviews - Natural Language Processing and Sentiment Analysis

import json
import os,sys

path = './hw2_data/review_train.json'
reviews = open(path,"r")
reviewDocs = []
for review in reviews:
    dataDict = json.loads(review)
    reviewLists = list(dataDict.values())
    reviewDocs.append(reviewLists)
print(len(reviewDocs))
#test set

testpath = './hw2_data/review_test.json'
testreviews = open(testpath,"r")
testreviewDocs = []
for testreview in testreviews:
    dataDict = json.loads(testreview)
    reviewLists = list(dataDict.values())
    testreviewDocs.append(reviewLists)
print(len(testreviewDocs)) 


# In[ ]:


#Using Document Based Embeddings

from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize

corpus = []
for doc in reviewDocs:
    corpus.append(doc[0])
    
for doc in testreviewDocs:
    corpus.append(doc[0])
    
tagged_data = [TaggedDocument(words=word_tokenize(_d.lower()), tags=[str(i)]) for i, _d in enumerate(corpus)]


# In[ ]:


# Training the doc2vec

max_epochs = 10
vec_size = 20
alpha = 0.025

model = Doc2Vec(size=vec_size,
                alpha=alpha, 
                min_alpha=0.00025,
                min_count=1,
                dm =1)
  
model.build_vocab(tagged_data)

for epoch in range(max_epochs):
    print('iteration {0}'.format(epoch))
    model.train(tagged_data,
                total_examples=model.corpus_count,
                epochs=model.iter)
    # decrease the learning rate
    model.alpha -= 0.0002
    # fix the learning rate, no decay
    model.min_alpha = model.alpha

model.save("d2v.model")
print("Model Saved")


# In[ ]:


#using embeddings to represent each review

corpus = []
for doc in reviewDocs:
    corpus.append(doc[0])
X = []
for i in range(len(corpus)):
    X.append(list(model[i]))
    
X = np.array(X)
print(X.shape)

testcorpus = []

for doc in testreviewDocs:
    testcorpus.append(doc[0])
X_test = []

for i in range(len(corpus),len(testcorpus)+len(corpus)):
    X_test.append(list(model[i]))
X_test = np.array(X_test)
print(X_test.shape)


# In[ ]:


y = [] # the labels on the training data
for doc in reviewDocs:
    y.append(doc[2])
    
y = np.array(y)
print(y.shape)

y_test = [] # labels on test data
for doc in testreviewDocs:
    y_test.append(doc[2])
y_test = np.array(y_test)
print(y_test.shape)


# In[ ]:


#Train a SVM
from sklearn import svm

clf = svm.SVC(gamma='scale')
y = y.reshape(-1,1)
clf.fit(X, y) 

y_predict = clf.predict(X_test)

