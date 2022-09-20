# -*- coding: utf-8 -*-
"""
Created on Sun Sep 18 18:23:10 2022

@author: omars
This implementation is the from scratch implementation for the celtic
mutation problem.
"""
import copy
from itertools import combinations

def load_file(file_name):
    #load file
    file = open(f'{file_name}', 'r', encoding = "UTF-8").readlines()
    return file

def create_train_dev_sets(file):
#create train and dev sets
    words = []
    x_train = []
    y_train = []
    x_dev = []
    y_dev = []

    for line in file:
        word = line.strip().split("\t")
        words.append(word)

    # find first sentence after 80% datapoints mark
    sentences = words[int(len(words)*0.8):]
    for sentence in sentences:
        if sentence[0] == "<S>":
            index = sentences.index(sentence)
            index += 1
            break

    for i in range(int(len(words)*0.8+index)):
        x_train.append(words[i][0])
        y_train.append(words[i][1])
    for i in range(int(len(words)*0.8+index), len(words)):
        x_dev.append(words[i][0])
        y_dev.append(words[i][1])

    return x_train, y_train, x_dev, y_dev

def create_hmm_dict(x_train, y_train):
    tags = {}
    count_dict = {}
    for i in range(len(x_train)):
        if x_train[i] in tags.keys():
            if y_train[i] in tags[x_train[i]].keys():
                tags[x_train[i]][y_train[i]] += 1
            else:
                tags[x_train[i]][y_train[i]] = 1
        else:
            tags[x_train[i]] = {}
            tags[x_train[i]][y_train[i]] = 1
        if y_train[i] in count_dict.keys():
            count_dict[y_train[i]] += 1
        else:
            count_dict[y_train[i]] = 1

    return tags, count_dict

def create_frequency_dict(tags_dict, count_dict):
    tags_frequency_dict = copy.deepcopy(tags_dict)
    for key in tags_frequency_dict.keys():
        for key2 in tags_frequency_dict[key].keys():
            tags_frequency_dict[key][key2] = tags_frequency_dict[key][key2]/ count_dict[key2]

    return tags_frequency_dict, count_dict

def create_cobigram_dict(labels):
    bigram_dict = {}
    for i in range(len(y_train)-1):
        if labels[i] in bigram_dict.keys():
            if labels[i+1] in bigram_dict[labels[i]].keys():
                bigram_dict[labels[i]][labels[i+1]] += 1
            else:
                bigram_dict[labels[i]][labels[i+1]] = 1
        else:
            bigram_dict[labels[i]] = {}
            bigram_dict[labels[i]][labels[i+1]] = 1

    return bigram_dict

def create_bigram_frequency_dict(bigram_dict):
    bigram_frequency_dict = copy.deepcopy(bigram_dict)
    for key in bigram_frequency_dict.keys():
        for key2 in bigram_frequency_dict[key].keys():
            bigram_frequency_dict[key][key2] = bigram_frequency_dict[key][key2]/ count_dict[key]

    return bigram_frequency_dict

def create_sentence_start_dict(sentences, labels):
    sentence = []
    label = []
    freq_dict = {}
    sentence_count = 0
    for i in range(len(sentences)):
      if sentences[i]=='<S>':
        sentence_count += 1
        if label[0] in freq_dict.keys():
            freq_dict[label[0]] += 1
        else:
            freq_dict[label[0]] = 1
        sentence = []
        label = []
      else:
        sentence.append(sentences[i])
        label.append(labels[i])
    for key in freq_dict.keys():
        freq_dict[key] = freq_dict[key]/ sentence_count
    return freq_dict


def predict_from_scratch(sentence, labels, frequency_dict, bigram_dict, sentence_dict):
    # print(sentence)
    viterbi(sentence, labels, frequency_dict, bigram_dict, sentence_dict)
    sequence_of_tags = []
    # for word in sentence:
    #     sequence_of_tags.append(list(frequency_dict[word].keys()))
    # print(sequence_of_tags)

def viterbi(words, tags, frequency_dict, bigram_dict, sentence_dict):
    V = {}
    B = {}
    tag_list = sentence_dict.keys()
    for t in tag_list:
        try:
            V[(t,0)] = sentence_dict[t]*frequency_dict[words[0]][t]
        except:
            V[(t,0)] = sentence_dict[t]*0
        print(V)

def evaluate(words, tags, frequency_dict, bigram_dict, sentence_dict):
    sentence = []
    labels = []
    for i in range(len(words)):
      if words[i]=='<S>':
        predict_from_scratch(sentence, labels, frequency_dict, bigram_dict, sentence_dict)
        sentence = []
        labels = []
      else:
        sentence.append(words[i])
        labels.append(tags[i])

if __name__ == "__main__":
    file_name = 'train.tsv'

    file = load_file(file_name)

    x_train, y_train, x_dev, y_dev = create_train_dev_sets(file)

    tags_dict, count_dict = create_hmm_dict(x_train, y_train)

    tags_frequency_dict, count_dict = create_frequency_dict(tags_dict, count_dict)

    bigram_dict = create_cobigram_dict(y_train)

    bigram_frequency_dict = create_bigram_frequency_dict(bigram_dict)

    sentence_freq_dict = create_sentence_start_dict(x_train, y_train)

    evaluate(x_train, y_train, tags_frequency_dict, bigram_frequency_dict, sentence_freq_dict)

    test = [['N', 'S', 'U'], ['N'], ['N'], ['N'], ['N'], ['N'], ['N'], ['N', 'S'], ['N', 'S', 'U'], ['S', 'N'], ['S', 'N'], ['N', 'S', 'U'], ['N', 'S', 'H', 'U'], ['S', 'U', 'N'], ['N'], ['N'], ['N'], ['N', 'U', 'H', 'T'], ['N'], ['N'], ['N', 'H', 'U'], ['N', 'H', 'T', 'U', 'S'], ['S', 'N', 'T'], ['N', 'S'], ['N'], ['S', 'N', 'U'], ['N', 'S', 'U'], ['N', 'H'], ['N', 'U'], ['N'], ['N', 'S'], ['S', 'N', 'U'], ['N', 'H'], ['N', 'S'], ['N', 'U', 'S', 'H', 'T'], ['N'], ['N'], ['H', 'N'], ['N'], ['N'], ['N'], ['N'], ['N'], ['N', 'H', 'T'], ['N'], ['N'], ['N'], ['N'], ['N'], ['N', 'U'], ['N'], ['N'], ['N'], ['N'], ['N', 'H', 'T'], ['S', 'U'], ['N'], ['N'], ['N'], ['N'], ['N']]
    num_of_comb = 1
    for i in test:
        num_of_comb *= len(i)


