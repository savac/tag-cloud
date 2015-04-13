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
    num_clusters = (len(tags)+1)/4

    # Initalize a k-means object and use it to extract centroids
    print "Running K means"
    '''
    kmeans_clustering = KMeans( n_clusters = num_clusters)
    kmeanFit = kmeans_clustering.fit( word_vectors )
    idx = kmeanFit.labels_
    centers = kmeanFit.cluster_centers_
    '''
    vecabular_size = len(model.index2word)

    word_sub_vectors = []
    sub_vecab = []
    for i in range(0, vecabular_size):
        word = model.index2word[i]
        for tag in tags:
            if tag.lower() == word:
                word_sub_vectors.append(word_vectors[i])
                sub_vecab.append(word)

    kmeans_clustering = KMeans( n_clusters = num_clusters)
    kmeanFit = kmeans_clustering.fit( word_sub_vectors )
    idx = kmeanFit.labels_

    # Create a Word / Index dictionary, mapping each vocabulary word to
    # a cluster number
    word_centroid_map = dict(zip( sub_vecab, idx ))


    clusterDist = kmeans_clustering.transform( word_vectors )

    '''
    cluster_tags = []
    cluster_id = []
    for tag in tags:
        if tag.lower() in word_centroid_map:
            id = word_centroid_map[tag.lower()]
            cluster_id.append(id)
            cluster_tags.append(model.index2word[np.argmin(clusterDist[:,id])])
    '''

    tag_clusters = [[] for x in xrange(num_clusters)]
    for tag in tags:
        if tag.lower() in word_centroid_map:
            id = word_centroid_map[tag.lower()]
            tag_clusters[id].append(tag)


    return tag_clusters

#    if __name__ == '__main__':




