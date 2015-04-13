#### Tag-cloud for the speeches of US presidents

This is a project that generates a tag cloud for important speeches made by US Presidents. The pre-requisites for running the code in this repo are:
Python 2.7.x<br>
NumPy<br>
NLTK<br>
Gensim<br>
Scikit-Learn<br>

All code needs to be run from directoy code/ using the Python command line or a suitable wrapper such as iPython.<br>

Import modules:<br>
```
import utils
import models
import clustering
import utils_word2vec
```

Read the entire corpus to obtain a matrix of the size <number of documents> x <vocabulary size> that contains the number of occurences of each vocabulary term in each document. We also obtain the vocabulary.<br>
```
(vocab,tf) = utils.read_corpus(1, False) # use ngram order of 1, don't weight words depending on position so each word occurence has a weigh of 1
```

Use a one of the models to assign weights to each of the words in every document:<br>
```
weights = models.tfidf(tf) # get the TD*IDF weights
```

Let's print the tags for the first document in info.tsv which Barack Obama's Acceptance Speech for the Nomination from the Democrat party:<br>
```
topWords = utils.get_tags(vocab, weights, 0)
print topWords
```

Before we can cluster the tags we need to create a vector representation for the words in our vocabulary. We use the word2vec tool and train on our collection of president's speeches. It's also possible train on any other large corpus as we are only trying to find out what words appear close to each other:<br>
'''
utils_word2vec.write_model_word2vec() # this will create a word2vec model and save it on the disk
'''

Now we can cluster the tags found out earlier in groups so we can get a better idea what topics are discussed in the speech:<br>
```
topClusters = clustering.run_clustering(topWords)
```



