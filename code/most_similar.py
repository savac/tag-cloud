__author__ = 'kolenka'

import utils
import gensim
import sklearn
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
import numpy as np
import os
import nltk.data
import sys

import utils_word2vec

def get_similar(tags):
    # vector of tags acheaved from other models
    # tags = ['muslim','holy']
    print "Reading word2vec model"
    model = utils_word2vec.read_word2vec()

    similar_tags = []
    for tag in tags:
        new_tag = model.most_similar(positive = tag.split(), topn = 1)
        similar_tags.append(new_tag)

    return similar_tags

if __name__ == '__main__':
    topWords = ['believe Americans', 'dreams hopes', 'hopes goals', 'believe fate', 'Martin Treptow', 'fate fall', 'act worthy', 'fall us', 'happiness liberty', 're sick', 'freedom those', 'provide opportunity', 'no misunderstanding', 'go away', 'growth government', 'us renew', 'few us', 'look answer', 'each Inaugural', 'taken aimed']
    print "Similar tags"
    print get_similar(topWords)



