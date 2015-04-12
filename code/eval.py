import nltk
import numpy as np
import models
import utils

def get_actual_tags(path):
    tags = []
    filepath = "../eval/" + path
    with open(filepath, "r") as content:
        for line in content:
            tags.append(line.strip())
    return tags

def precision(predicted_tags, actual_tags):
    rel_tags = set(actual_tags)
    ret_tags = set(predicted_tags)
    numerator = len(rel_tags & ret_tags)
    denominator = len(ret_tags)
    if denominator == 0:
        return 0.0
    else:
        return float(numerator)/denominator

def recall(predicted_tags, actual_tags):
    rel_tags = set(actual_tags)
    ret_tags = set(predicted_tags)
    numerator = len(rel_tags & ret_tags)
    denominator = len(rel_tags)
    if denominator == 0:
        return 0.0
    else:
        return float(numerator)/denominator

def average_precision(extracted_tags):
    speech_info = utils.read_speech_info()
    N = len(speech_info)
    precision_sum = 0.0
    for i in range(N):
        path = speech_info[i]
        predicted_tags = [t.lower() for t in extracted_tags[path]]
        actual_tags = get_actual_tags(path)
        p = precision(predicted_tags, actual_tags)
        precision_sum += p
    return precision_sum/N

def average_recall(extracted_tags):
    speech_info = utils.read_speech_info()
    N = len(speech_info)
    recall_sum = 0.0
    for i in range(N):
        path = speech_info[i]
        predicted_tags = [t.lower() for t in extracted_tags[path]]
        actual_tags = get_actual_tags(path)
        r = recall(predicted_tags, actual_tags)
        recall_sum += r
    return recall_sum/N

def compute_pr(model):
    '''Takes in a model to produce tags, then evaluates PR of model'''
    print "Evaluating %s" % (model)
    (vocab, tf) = utils.read_corpus()
    top_words = models.run_model(model, vocab, tf)
    avg_recall = average_recall(top_words)
    avg_precision = average_precision(top_words)
    print "Average Precision: %f" % (avg_precision)
    print "Average Recall: %f" % (avg_recall)

if __name__ == '__main__':
    compute_pr("tfidf")