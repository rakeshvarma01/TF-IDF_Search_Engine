# TF-IDF_Search_Engine
# Chirag Shah


This folder consists of :

1. TF_IDF_SE.py - handles all the assignment methods and implemenations.
2. presidential_debates folder as the corpus which is being in current implementation.
3. nltk_data folder that has the stopwords.
4. A tf_idf.txt that outputs weight of each word w.r.t file provided.

query(qstring): returns a tuple in the form of (filename of the document, score), where the document is the query answer with respect to "qstring" . 
If no document contains any token in the query, returns ("None",0). 

* getidf(token): returns the inverse document frequency of a token. If the token doesn't exist in the corpus, returns -1. The parameter 'token' is already stemmed.
 (It means you should not perform stemming inside this function.) Note the differences between getidf("hispan") and getidf("hispanic"). 

* getweight(filename,token): returns the TF-IDF weight of a token in the document named 'filename'. If the token doesn't exist in the document, returns 0. 
The parameter 'token' is already stemmed. (It means you should not perform stemming inside this function.) Note that both getweight("1960-10-21.txt","reason") 
and getweight("2012-10-16.txt","hispanic") return 0, but for different reasons. 

Note:
Current path for corpusroot: corpusroot = './presidential_debates'
The implementation returns documents most relevant to input query based on TF – IDF weight statistics and computing cosine similarity within 3 seconds.

