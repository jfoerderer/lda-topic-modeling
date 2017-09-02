# -*- coding: utf-8 -*-

import csv
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
from gensim import corpora
import gensim
import os
import re
from nltk.tokenize import RegexpTokenizer

#SET PATH
path = r''
inputname=""

def remove_html_tags(text):
        """Remove html tags from a string"""
     
        clean = re.compile('<.*?>')
        return re.sub(clean, '', text)
        
#setup
tokenizer = RegexpTokenizer(r'\w+')
en_stop = get_stop_words('en')
p_stemmer = PorterStemmer()

 
fn = os.path.join(path, inputname)
doc_set = []

with open(fn, encoding="utf8" ) as f:
            csv_f = csv.reader(f)
            for i, row in enumerate(csv_f):
               if i > 1 and len(row) > 1 :
                
                 temp=remove_html_tags(row[1]) 
                 temp = re.sub("[^a-zA-Z ]","", temp)
                 doc_set.append(temp)
                 
texts = []

for i in doc_set:
    
    if i.strip():
        raw = i.lower()           
        tokens = tokenizer.tokenize(raw)
        if len(tokens)>5:
            stopped_tokens = [i for i in tokens if not i in en_stop]
            texts.append(stopped_tokens)

dictionary = corpora.Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]

lsi = gensim.models.lsimodel.LsiModel(corpus, id2word=dictionary, num_topics=5  )
print (lsi.print_topics(num_topics=3, num_words=3))

 ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=20, id2word = dictionary, passes=20)
 print(ldamodel.print_topics(num_topics=20, num_words=5))
 K = ldamodel.num_topics
 topicWordProbMat = ldamodel.print_topics(K)
