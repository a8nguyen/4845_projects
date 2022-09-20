# -*- coding: utf-8 -*-
"""
Created on Sun Sep 18 18:23:10 2022

@author: omars
This implementation is the from scratch implementation for the celtic
mutation problem.
"""
import copy
import pickle

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
    index = 0
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

def create_cooccurence_dict(labels):
    co_occurence_dict = {}
    for i in range(len(y_train)-1):
        if labels[i] in co_occurence_dict.keys():
            if labels[i+1] in co_occurence_dict[labels[i]].keys():
                co_occurence_dict[labels[i]][labels[i+1]] += 1
            else:
                co_occurence_dict[labels[i]][labels[i+1]] = 1
        else:
            co_occurence_dict[labels[i]] = {}
            co_occurence_dict[labels[i]][labels[i+1]] = 1

    return co_occurence_dict

def create_co_occurence_frequency_dict(co_occurence_dict):
    co_occurence_frequency_dict = copy.deepcopy(co_occurence_dict)
    for key in co_occurence_frequency_dict.keys():
        for key2 in co_occurence_frequency_dict[key].keys():
            co_occurence_frequency_dict[key][key2] = co_occurence_frequency_dict[key][key2]/ count_dict[key]

    return co_occurence_frequency_dict

if __name__ == "__main__":
    file_name = 'train.tsv'

    file = load_file(file_name)

    x_train, y_train, x_dev, y_dev = create_train_dev_sets(file)

    tags_dict, count_dict = create_hmm_dict(x_train, y_train)

    tags_frequency_dict, count_dict = create_frequency_dict(tags_dict, count_dict)

    co_occurence_dict = create_cooccurence_dict(y_train)

    co_occurence_frequency_dict = create_co_occurence_frequency_dict(co_occurence_dict)


