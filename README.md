#### Tag-cloud for the speeches of American presidents

Run as following from code/:<br>
```
    import pspeech
    (vocab,tf) = pspeech.read_corpus() # read in collection and count tf
    weights = pspeech.tfidf(tf) # get the TD*ODF weights
    topWords = pspeech.get_tags(vocab, weights, 0) # get top 20 words for Obama's Acceptance Speech
    print topWords
```
