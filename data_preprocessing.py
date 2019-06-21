# encoding=utf8
import jsonlines
import pickle
import numpy as np

data_size = 10000

sentence_max_length = 200
count = 1
word2id = {}
id2word = {}
tag2id = {'b': 1, 'm': 2, 'e': 3, 's': 4, 'padding': 0}
id2tag = {}
for key, val in tag2id.items():
    id2tag[val] = key

count_size = 0
with open("corpus/train_data.json", "r+", encoding="utf8") as f:
    for item in jsonlines.Reader(f):
        if count_size < data_size:
            count_size += 1
        else:
            break
        text = item["text"]
        for word in text:
            if word not in word2id:
                word2id[word] = count
                count += 1
            else:
                pass
f.close()
for key, val in word2id.items():
    id2word[val] = key

count_size = 0
with open("corpus/train_data.json", "r+", encoding="utf8") as f:
    raw_train_data = []
    raw_train_label = []
    for item in jsonlines.Reader(f):
        if count_size < data_size:
            count_size += 1
        else:
            break
        sentence_data = []
        sentence_label = []
        postag = item["postag"]
        text = item["text"]
        len_1 = 0
        len_2 = len(text)
        for pos in postag:
            word = pos["word"]
            len_1 += len(word)
        if len_1 == len_2:
            for pos in postag:
                word = pos["word"]
                if len(word) == 1:
                    sentence_data.append(word)
                    sentence_label.append('s')
                else:
                    for i in range(len(word)):
                        if i == 0:
                            sentence_data.append(word[i])
                            sentence_label.append('b')
                        if i == len(word) - 1:
                            sentence_data.append(word[i])
                            sentence_label.append('e')
                        if (i != 0) and (i != len(word) - 1):
                            sentence_data.append(word[i])
                            sentence_label.append('m')
        if sentence_data:
            raw_train_data.append(sentence_data)
            raw_train_label.append(sentence_label)
f.close()

train_data = []
train_label = []
for i in range(len(raw_train_data)):
    sentence_data = []
    sentence_label = []
    for j in range(len(raw_train_data[i])):
        word = raw_train_data[i][j]
        sentence_data.append(word2id[word])
        tag = raw_train_label[i][j]
        sentence_label.append(tag2id[tag])
    if len(sentence_data) > sentence_max_length:
        sentence_data = sentence_data[:sentence_max_length]
        sentence_label = sentence_label[:sentence_max_length]
    else:
        for k in range(sentence_max_length - len(sentence_data)):
            sentence_data.append(0)
            sentence_label.append(0)
    train_data.append(sentence_data)
    train_label.append(sentence_label)

np.array(train_data)
np.array(train_label)

dict_path = 'data/your_dict.pkl'
with open(dict_path, 'wb') as inp:
    pickle.dump(word2id, inp)
    pickle.dump(id2word, inp)
    pickle.dump(tag2id, inp)
    pickle.dump(id2tag, inp)
inp.close()

train_path = 'data/your_train_data.pkl'
with open(train_path, 'wb') as inp:
    pickle.dump(train_data, inp)
    pickle.dump(train_label, inp)
inp.close()