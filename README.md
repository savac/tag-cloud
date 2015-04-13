#### Tag-cloud for the speeches of US presidents
This is a project that generates a tag cloud for important speeches made by US Presidents. The speeches have been uploaded to directory data/. The tab-separated file info.tsv in the head directory contains the details about the speeches such as the president's name, speech title and date.<br>

The prerequisites for running the code in this repo are:<br>
* Python 2.7.x<br>
* NumPy<br>
* NLTK<br>
* Gensim<br>
* Scikit-Learn<br>

All code needs to be run from directory code/ using the Python command line or a suitable wrapper such as iPython.<br>

Import modules:<br>
```
import utils
import models
import clustering
import utils_word2vec
```

Read the entire corpus to obtain a matrix of the size (number of documents) x (vocabulary size) that contains the number of occurences of each vocabulary term in every document. We also obtain the vocabulary.<br>
```
(vocab,tf) = utils.read_corpus(1, False) # use ngram order of 1, don't weight words depending on position so each word occurence has a weigh of 1
```

Use one of the models to assign weights to each of the words in every document:<br>
```
weights = models.tfidf(tf) # get the TD*IDF weights
```

Let's print the tags for the first document in info.tsv which is Barack Obama's Acceptance Speech for the Nomination from the Democrat party:<br>
```
topWords = utils.get_tags(vocab, weights, 0)
print topWords
```
Result:<br>
```
['McCain',
 'promise',
 'John',
 'Bush',
 'whiners',
 'Iraq',
 'Democrats',
 'keep',
 'jobs',
 'George',
 'renewable',
 'tough',
 'america',
 'tonight',
 'Clinton',
 'cars',
 'change',
 'politics',
 'Republicans',
 'veterans']
```
<br>


So far we have obtained the tags for a given document. As a more advanced task We can cluster (or categorise) the tags to get a deeper meaning of the topics discussed in the speech. However before we can cluster the tags we need to create a vector representation for the words in our vocabulary. We use the word2vec tool and train on our collection of president's speeches. It's also possible train on any other large corpus as we are only trying to find out what words appear close to each other:<br>
```
utils_word2vec.write_model_word2vec() # this will create a word2vec model and save it on the disk
```

Now we can cluster the tags found out earlier so we can get a better idea what topics are discussed in the given speech:<br>
```
topClusters = clustering.run_clustering(topWords)
```
Results:<br>
```
[['tonight'],
 ['John',  'Bush',  'whiners',  'Iraq',  'Democrats',  'George',  'renewable',  'america',  'Clinton',  'cars',  'politics',  'Republicans',  'veterans'],
 ['jobs'],
 ['promise', 'keep', 'change'],
 ['tough']]

```
We can see two distinct topics emerging. Barack Obama,the Democratic Party nominee at the time, is discussing the his opponent and the current presindent in in one of the topics. The topic containing 'promise', 'keep' and 'change' is related to his intentions if he became president.

