import nltk
import numpy as np

def precision(predicted_tags, tags):
    rel_tags = 0.0
    ret_tags = len(predicted_tags)
    for pt in predicted_tags:
        if pt in tags:
            rel_tags += 1.0
    return rel_tags/ret_tags

def recall(predicted_tags, tags):
    rel_tags = 0.0
    ret_tags = len(predicted_tags)
    for pt in predicted_tags:
        if pt in tags:
            rel_tags += 1.0
    return ret_tags/rel_tags

def avg_precison(speech_tags, actual_tags):
    precision_sum = 0.0
    for i in range(len(speech_tags)):
        p = precision(speech_tags[i], actual_tags[i])
        precision_sum += p
    return precision_sum/len(speech_tags)

def avg_precison(speech_tags, actual_tags):
    recall_sum = 0.0
    for i in range(len(speech_tags)):
        r = recall(speech_tags[i], actual_tags[i])
        precision_sum += r
    return recall_sum/len(speech_tags)