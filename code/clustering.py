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

def run_clustering(tags):
    # vector of tags acheaved from other models
    # tags = ['muslim','holy']
    print "Reading word2vec model"
    model = utils_word2vec.read_word2vec()
    word_vectors = model.syn0
    num_clusters = 2*len(tags) - 1

    # Initalize a k-means object and use it to extract centroids
    print "Running K means"
    kmeans_clustering = KMeans( n_clusters = num_clusters)
    kmeanFit = kmeans_clustering.fit( word_vectors )
    idx = kmeanFit.labels_
    centers = kmeanFit.cluster_centers_

    # Create a Word / Index dictionary, mapping each vocabulary word to
    # a cluster number
    word_centroid_map = dict(zip( model.index2word, idx ))


    clusterDist = kmeans_clustering.transform( word_vectors )

    cluster_tags = []
    cluster_id = []
    for tag in tags:
        if tag.lower() in word_centroid_map:
            id = word_centroid_map[tag.lower()]
            cluster_id.append(id)
            cluster_tags.append(model.index2word[np.argmin(clusterDist[:,id])])

    print "Bag of centroids tags:"
    print cluster_tags

    return cluster_tags

#    if __name__ == '__main__':




