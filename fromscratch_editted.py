# -*- coding: utf-8 -*-
"""
Created on Sun Sep 18 18:23:10 2022

@author: omars
This implementation is the from scratch implementation for the celtic
mutation problem.
"""
import copy
from collections import defaultdict

def load_file(file_name):
    #load file
    file = open(f'{file_name}', 'r', encoding = "UTF-8").readlines()
    return file


def create_train_dev_sets(file, proportions=0.8):
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
    sentences = words[int(len(words)*proportions):]
    for sentence in sentences:
        if sentence[0] == "<S>":
            index = sentences.index(sentence)
            index += 1
            break

    for i in range(int(len(words)*proportions+index)):
        x_train.append(words[i][0])
        y_train.append(words[i][1])
    for i in range(int(len(words)*proportions+index), len(words)):
        x_dev.append(words[i][0])
        y_dev.append(words[i][1])
    return x_train, y_train, x_dev, y_dev

def create_hmm_dict(x_train, y_train):
    #emission probability here!
    tags = defaultdict(lambda: defaultdict(int))  # tags[tag_i][word_i]
    count_dict = defaultdict(int)
    for i in range(len(x_train)):
        tags[x_train[i]][y_train[i]] += 1
        count_dict[y_train[i]] += 1
    return tags, count_dict


def create_frequency_dict(tags_dict, count_dict, alpha=0):
    #tags dict = tags[tag_i][word_i]
    tags_frequency_dict = defaultdict(lambda : defaultdict( lambda :alpha)) 
    for key in tags_dict:
        for key2 in tags_dict[key]:
            tags_frequency_dict[key][key2] = tags_dict[key][key2] /count_dict[key2]
            #print(tags_frequency_dict[key][key2])
    return tags_frequency_dict

def create_upto_ngram_dict(labels, n_precede = 1):
    #transition matrix stuff
    bigram_dict = defaultdict(lambda: defaultdict(int)) #n_gram[type_{i-2} type_{i-1}][ type_{i}]
    for i in range(len(labels)-n_precede):
        for n in range(1, n_precede+1):
            prev_phrase = ' '.join([w for w in labels[i: i+n]])
            bigram_dict[prev_phrase][labels[i+n_precede]] += 1
    return bigram_dict

def create_upto_ngram_frequency_dict(labels, count_dict):
    bigram_dict = defaultdict(lambda: defaultdict(lambda: alpha))
    for key in labels:
        for key2 in labels[key]:
            bigram_dict[key][key2] = labels[key][key2] / \
                count_dict[key]
            #print(tags_frequency_dict[key][key2])
    return bigram_dict




def create_sentence_start_dict(sentences, labels):
    sentence = []
    label = []
    freq_dict = defaultdict(float) #freq_dict[tag]
    sentence_count = 0
    for i in range(len(sentences)):
      if sentences[i]=='<S>':
        sentence_count += 1
        freq_dict[label[0]] += 1

        sentence = []
        label = []
      else:
        sentence.append(sentences[i])
        label.append(labels[i])
    for key in freq_dict.keys():
        freq_dict[key] = freq_dict[key]/ sentence_count
    return freq_dict


def predict_from_scratch(sentence, frequency_dict, bigram_dict, sentence_dict):
    tags = viterbi(sentence, frequency_dict, bigram_dict, sentence_dict)
    return tags


def viterbi(words,emission_vectors, transition_mat, sentence_dict):
    V = defaultdict(lambda : defaultdict(float))
    B = defaultdict(lambda : defaultdict(str))
    tag_list = sentence_dict.keys()

    for t in tag_list:
        V[0][t] = sentence_dict[t] * emission_vectors[words[0]][t]

    for i in range(1,len(words)):
        for t in tag_list:
            pair = argmax(V,tag_list,t,i, transition_mat)
            B[i][t] = pair[0]
            V[i][t] = pair[1]*emission_vectors[words[i]][t]
        

    final_labels = get_best_tag(words, V, B, tag_list)
    return final_labels

def argmax(V,tag_list,t,i, transition_mat):
    ans=-1
    best=None
    for s in tag_list:
        temp = V[i-1][s] * transition_mat[t][s]
        if temp > ans:
            ans = temp
            best = s
    return (best,ans)

def get_best_tag(sent, V,B, tags):
    best_ending = None
    best_max = -1

    for tag in tags:
        if V[len(sent) - 1][tag] > best_max:
            best_max = V[len(sent) - 1][tag]
            best_ending = tag
    seq = [best_ending]
    for i in reversed(range(1, len(sent))):
        prev = B[i][seq[-1]]
        #print( seq[-1])
        seq.append(prev)
    #print(len(seq), len(sent))
    return seq[::-1]


def evaluate(words,tags, frequency_dict, bigram_dict, sentence_dict):
    sentence = []
    labels = []
    final_tags = []
    correct = 0
    num = 0
    for i in range(len(words)):
      if words[i] == '<S>':
        print(i)
        num += 1
        final_tags.append(predict_from_scratch(
            sentence, frequency_dict, bigram_dict, sentence_dict))
        final_tags.append('N')
        sentence = []
        labels = []
      else:
        sentence.append(words[i])
        labels.append(tags[i])
    final_tag = [j for i in final_tags for j in i]
    for i in range(len(tags)):
        if final_tag[i] == tags[i]:
            correct += 1
    accuracy = correct / len(tags)
    return accuracy

if __name__ == "__main__":
    file_train = load_file('train.tsv')
    file_test = load_file('test.tsv')
    x_train, y_train, x_dev, y_dev = [], [], [], []
    for line in file_train:
        word = line.strip().split("\t")
        x_train.append(word[0])
        y_train.append(word[1])
    for line in file_test:
        word = line.strip().split("\t")
        x_dev.append(word[0])
        y_dev.append(word[1])

    word_to_tag_count, count_dict = create_hmm_dict(x_train, y_train)

    tag_tag_count = create_upto_ngram_dict(y_train)

    emission = create_frequency_dict(word_to_tag_count, count_dict, alpha = 0.0000001)

    tmat = create_upto_ngram_frequency_dict(tag_tag_count, count_dict)

    initial_prob_distrib = create_sentence_start_dict(x_train, y_train)
    print("YO")
   
    #training_accuracy = evaluate(
    #    x_train, y_train, tag_to_word, transition_matrix_tag_tag, initial_prob_distrib)
    dev_accuracy = evaluate(
        x_dev, y_dev,  emission, tmat, initial_prob_distrib)



