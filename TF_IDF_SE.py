"""
@author: Chirag Shah
"""

import os
from nltk.tokenize import RegexpTokenizer
#import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import time
import math
#General Code Variables Initialization
corpusroot = './presidential_debates'
tokenizer = RegexpTokenizer(r'[a-zA-Z]+')
stemmer = PorterStemmer()
stopwords_english = set(stopwords.words('english'))
word_doc_dict={}
word_doc_dict_tf_idf={}
dict_tf_idf={}
tf_idf_query_based={}
sortedPostingDic={}
num_of_doc = len(os.listdir(corpusroot))
print("Total Files: ",num_of_doc)

#return the inverse document frequency of a token
def getidf(token):
    d = word_doc_dict.get(token,0)
    if d!=0:
        numDocumentsWithThisToken = len(d)
        if numDocumentsWithThisToken > 0:
            idf = math.log10(num_of_doc / numDocumentsWithThisToken)
            return idf
        else:
            return -1
    else:
        return -1

#return the TF-IDF weight of a token in the document named 'filename'
def getweight(filename,token):
    #Get the overall Normalized weight..
    norm = word_doc_dict_tf_idf.get(filename,0)
    #calculate Term Frequency
    term_freq = word_doc_dict.get(token,{}).get(filename,0)
    #calculate Term Frequency * IDF 
    tf_idf = (1 + (math.log10(term_freq)))  * getidf(token.lower())
    #print("Just Tf-Idf",tf_idf)
    #get Final Weight of the token by dividing it with Norm
    norm_tf_idf = tf_idf/norm
    return norm_tf_idf

#return a tuple in the form of (filename of the document, score)
def query(qstring):
    tokensQuery = tokenizer.tokenize(qstring)
    tokensList={""}
    word_count = len(tokensQuery)
    #print("query words ",word_count)
    tf_q_norm=0
    for w in tokensQuery:
        if w not in stopwords_english:
            stem_words=stemmer.stem(w)
            tokensList.add(stem_words)
    tokensList.remove("")
    for k in tokensList:
        c=tokensQuery.count(k)
        if(c>0):ct = c
        else:ct=1
        tf_q_norm += (1+(math.log10(ct)))**2
    #print("Tf Query norm",tf_q_norm)
    #print("Length of tokenList",tokensList)
    queryVector=calcNorm(k,tokensQuery,tokensList,"Query",tf_q_norm)
    #print(tf_q)
    #Calling getTfIdfVector to get a TF-IDF vector of Query Tokens..
    getTfIdfVector(tokensQuery)
    finalDocList=[]
    for i in sortedPostingDic:
        for j in sortedPostingDic[i]:
            count=0
            for k in sortedPostingDic:
                if j in sortedPostingDic[k]:
                    count+=1
            if count == word_count and j not in finalDocList:
                finalDocList.append(j)
    #print("FinaList:",finalDocList)
    #Get the Cosine Value and print the highest one with its Document
    return getCosine(queryVector,finalDocList)
	
#Method to calculate Cosine for a given QueryVector and Documents
def getCosine(queryVector,finalDocList):
    cosineDictionary={}
    for i in finalDocList:
        cosine=0
        for j in queryVector:
            wtfq = queryVector[j]
            #print("term freq for",j,"is ",wtfq)
            for k in dict_tf_idf:
                if k == j:
                    for l in dict_tf_idf[k]:
                        if l == i:
                            wtfidfd=dict_tf_idf[k][l]
                            #print("doc fre for ",l,"is",wtfidfd)
            cosine+=wtfq*wtfidfd
            cosineDictionary[i]=cosine
    #print(cosineDictionary)
    maxCos=0
    document="None"
    for i in cosineDictionary:
        if cosineDictionary[i] > maxCos:
            maxCos = cosineDictionary[i]
            document = i
    return (document, maxCos)

#Method to calculate Normalized weight for a query document based on respective tokens            
def calcNorm(k,tokens,tokensList,operation,tf_q_norm):    
    if operation == "Query":
        queryVector={}
        for k in tokensList:
            c=tokens.count(k)
            if(c>0):ct = c
            else:ct=1
            tf_q = ((1+(math.log10(ct))))/math.sqrt(tf_q_norm)
            queryVector[k]=tf_q
        return queryVector

# Method to calculate a TF-IDFvetor based on tokens
def getTfIdfVector(tokensQuery):     
    for (filename) in os.listdir(corpusroot):
        file = open(os.path.join(corpusroot, filename), "r", encoding='UTF-8')
        doc = file.read()
        doc = doc.lower() 
        tokens = tokenizer.tokenize(doc)
        for words in tokens:#constructing tf_idf vector
            if words not in stopwords_english and words in tokensQuery:
                stem_words=stemmer.stem(words)
                norm = word_doc_dict_tf_idf.get(filename,0)
                term_freq = word_doc_dict.get(stem_words,{}).get(filename,0)
                tf_idf = (1 + (math.log10(term_freq)))  * getidf(stem_words.lower())
                norm_tf_idf = tf_idf/norm
                if dict_tf_idf.get(stem_words,0)==0:
                    dict_tf_idf[stem_words]={}
                    dict_tf_idf[stem_words][filename]=norm_tf_idf
                else:
                    dict_tf_idf[stem_words][filename]=word_doc_dict[stem_words].get(filename,0)
                    dict_tf_idf[stem_words][filename]=norm_tf_idf
    #print(dict_tf_idf)
    temp=[]
    postings=[]
    for i in dict_tf_idf:
        for j in dict_tf_idf[i]:
            temp.append(dict_tf_idf[i][j])
        temp.sort(reverse=True)
        temp=temp[:10]
        for k in temp:
            for j in dict_tf_idf[i]:
                if dict_tf_idf[i][j]==k:
                    postings.append(j)
        sortedPostingDic[i] = postings
        temp=[]
        postings=[]
    #print(sortedPostingDic)
    file.close()   

#Method to Pre-process corpus in relevant dictionary...A Main method can not be skipped.
def preProcessCorpus():      
    for (filename) in os.listdir(corpusroot):
        file = open(os.path.join(corpusroot, filename), "r", encoding='UTF-8')
        doc = file.read()
        doc = doc.lower() 
        tokens = tokenizer.tokenize(doc)
        tokensList={""}#making a set for non-repetitive words to get calculate normalized weight
        for w in tokens:
            if w not in stopwords_english:
                stem_word=stemmer.stem(w)
                if word_doc_dict.get(stem_word,0)==0:
                    word_doc_dict[stem_word]={}
                    word_doc_dict[stem_word][filename]=1
                    tokensList.add(stem_word)
                else:
                    word_doc_dict[stem_word][filename]=word_doc_dict[stem_word].get(filename,0)
                    word_doc_dict[stem_word][filename]+=1
                    tokensList.add(stem_word)
        tokensList.remove("")#removing emtpy elements if any..
        #word_doc_dict_tf_idf[filename] = calcNorm(stem_word,tokens,tokensList,"Document","")
    file.close()

#Method to calculate Normalized weight for complete set of individual documents..
def getNorm():
     for (filename) in os.listdir(corpusroot):
        file = open(os.path.join(corpusroot, filename), "r", encoding='UTF-8')
        doc = file.read()
        doc = doc.lower() 
        tokens = tokenizer.tokenize(doc)
        tokensList={""}#making a set for non-repetitive words to get calculate normalized weigh
        for w in tokens:
            if w not in stopwords_english:
                stem_word=stemmer.stem(w)
                tokensList.add(stem_word)
        tokensList.remove("")#removing emtpy elements if any..
        tfidf=0
        for k in tokensList:
            c=tokens.count(k)
            if(c>0):ct = c
            else:ct=1
            ctif = getidf(k.lower())
            #x = ((1 + (math.log10(ct))) * ctif)
            tfidf += ((1 + (math.log10(ct))) * ctif)**2
        k=math.sqrt(tfidf)
        word_doc_dict_tf_idf[filename] = k
     file.close()

preProcessCorpus()
getNorm()
#getNorm()   
#start_time1 = time.time()
print("(%s, %.12f)" % query("terror attack"))
print("%.12f" % getidf("health"))
print("%.12f" % getweight("2012-10-03.txt","health"))
#print ((time.time() - start_time1), "seconds")