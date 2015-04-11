import nltk
import numpy as np
import sys

def read_info(infile="../info.tsv"):
    '''Read in info.tsv and output a list of lists of the form:
    [[doc_path, president, title, data],...]'''
    result = []
    with open(infile, "r") as content:
        for line in content:
            line = line.strip("\n")
            result.append(line.split("\t"))
    return result
    
def read_corpus(ngramOrder=1, posWeightFlag=False):
    '''
    Read all files in collection and return the tf count. The output array
    has a shape: nr_of_docs x vocabulary_size. This function also returns
    the vocalubary which is a list of words.
    '''
    info = read_info() # [[doc_path, president, title, data],...]
    NUM_DOCS = len(info) # get number of docs
    vocab = dict() # a list for the vocabulary
    thisList = list() # a list of document tf vectors
    indDict = 0
    for indDoc in range(0, NUM_DOCS):
        tokens = speech_tokenizer(info[indDoc][0]) # open speech file and tokenize
        this = [0]*len(vocab) # init with the current size of the vocab
        lengthDoc = float(len(tokens))
        for t in range(0, len(tokens)-ngramOrder+1):
            # Elena's idea: weight the counts depending where in the document the word is.
            if posWeightFlag:
                incVal = 0.5 + np.abs(t/lengthDoc-0.5) # 1 for beginning/end, 0.5 for middle
            else:
                incVal = 1
            new_token = ' '.join(tokens[t:t+ngramOrder])
            try:
                ind = vocab[new_token]
                this[ind]+=incVal
            except KeyError:
                vocab[new_token]=indDict
                indDict+=1
                this.append(incVal) # grow the document vector

        thisList.append(this)
        # progress
        sys.stdout.write("\rReading collection: %d%%" %(indDoc*100/NUM_DOCS))
        sys.stdout.flush()     
           
    # put the document vectors into a numpy array
    tf = np.zeros((NUM_DOCS, len(vocab)), dtype='uint8') # assume no ngram count exceeds 2^8
    for indDoc in range(0, NUM_DOCS):
        thisLength = len(thisList[indDoc])
        tf[indDoc, 0:thisLength] = thisList[indDoc]
    
    # will be using vocabulary with indexing later so faster in a list
    temp = ['']*len(vocab)
    for k in vocab:
        temp[vocab[k]]=k
    vocab=temp
    
    return (vocab, tf)

def speech_tokenizer(infile):
    
    '''
    common100 = {1: 'the', 2: 'of', 3: 'and', 4: 'a', 5: 'to', 6: 'in', 7: 'is', 8: 'you', 9: 'that', 10: 'it', 
    11: 'he', 12: 'was', 13: 'for', 14: 'on', 15: 'are', 16: 'as', 17: 'with', 18: 'his', 19: 'they', 
    20: 'I', 21: 'at', 22: 'be', 23: 'this', 24: 'have', 25: 'from', 26: 'or', 27: 'one', 28: 'had', 29: 'by', 
    30: 'word', 31: 'but', 32: 'not', 33: 'what', 34: 'all', 35: 'were', 36: 'we', 37: 'when', 38: 'your', 39: 'can', 
    40: 'said', 41: 'there', 42: 'use', 43: 'an', 44: 'each', 45: 'which', 46: 'she', 47: 'do', 48: 'how', 49: 'their', 
    50: 'if', 51: 'will', 52: 'up', 53: 'other', 54: 'about', 55: 'out', 56: 'many', 57: 'then', 58: 'them', 59: 'these', 
    60: 'so', 61: 'some', 62: 'her', 63: 'would', 64: 'make', 65: 'like', 66: 'him', 67: 'into', 68: 'time', 69: 'has', 
    70: 'look', 71: 'two', 72: 'more', 73: 'write', 74: 'go', 75: 'see', 76: 'number', 78: 'way', 79: 'could', 
    80: 'people', 81: 'my', 82: 'than', 83: 'first', 84: 'water', 85: 'been', 86: 'call', 87: 'who', 88: 'oil', 89: 'its', 
    90: 'now', 91: 'find', 92: 'long', 93: 'down', 94: 'day', 95: 'did', 96: 'get', 97: 'come', 98: 'made', 99: 'may', 100: 'part'}
    '''
    common100 = {1: 'the', 2: 'of', 3: 'and', 4: 'a', 5: 'to', 6: 'in', 7: 'is', 8: 'you', 9: 'that', 10: 'it', 
    11: 'he', 13: 'for', 14: 'on', 15: 'are', 16: 'as', 17: 'with', 18: 'his', 19: 'they', 
    20: 'I', 21: 'at', 22: 'be', 23: 'this', 25: 'from', 26: 'or', 27: 'one', 29: 'by', 
    30: 'word', 31: 'but', 32: 'not', 33: 'what', 34: 'all', 35: 'were', 36: 'we', 37: 'when', 38: 'your', 39: 'can', 
     41: 'there', 43: 'an', 45: 'which', 46: 'she', 47: 'do', 48: 'how', 49: 'their', 
    50: 'if', 51: 'will', 52: 'up', 53: 'other', 54: 'about', 55: 'out', 56: 'many', 57: 'then', 58: 'them', 59: 'these', 
    60: 'so', 61: 'some', 62: 'her', 63: 'would', 66: 'him', 67: 'into', 69: 'has', 
    82: 'than', 85: 'been', 87: 'who', 89: 'its', 90: 'Bless'}
    
    common100 = common100.values()
    
    pattern = r'''(?x)               # set flag to allow verbose regexps
              ([A-Z]\.)+         # abbreviations, e.g. U.S.A.
              | \$?\d+(\.\d+)?%? # numbers, incl. currency and percentages
              | \w+([-']\w+)*    # words w/ optional internal hyphens/apostrophe
              | [+/\-@&*]        # special characters with meanings
            '''
    tokenizer = nltk.tokenize.RegexpTokenizer(pattern) # this will only keep words

    # read file
    text = open(infile, 'r').read()
    
    # split into sentences and then change the first word to lower case
    sentences = nltk.sent_tokenize(text)
    tokens = []
    for sentence in sentences:
        this = tokenizer.tokenize(sentence)
        if len(this)>0:
            this[0] = this[0].lower()
        tokens = tokens + this
    
    # Disregard the most commom words in English
    res = []
    for token in tokens:
        if not(common100.count(token)):
            res.append(token)
    tokens = res
    
    return tokens

def filter_speech(filt, filt_type='president'):
    info = read_info()
    result = []
    for indDoc in range(0, len(info)):
        if filt_type=='president' and info[indDoc][1].lower().count(filt.lower()):
            result.append(indDoc)
        if filt_type=='year' and info[indDoc][3].lower().count(filt.lower()):
            result.append(indDoc)
    return result
            
def get_tags(vocab, weights, indDoc, topN=20):
    '''
    Given a vocabulary and weights return top N words for the document 
    with indec indDoc
    '''
    sortW = weights[indDoc,:].argsort()
    result = [vocab[i] for i in reversed(sortW[-topN:])]
    return result
    
    
if __name__ == '__main__':
    (vocab,tf) = read_corpus()
    weights = tfidf(tf)
    topWords = get_tags(vocab, weights, 0) # get top 20 words for Obama's Acceptance Speech
    print topWords

