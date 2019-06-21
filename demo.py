import pickle

dict_path = 'data/your_dict.pkl'
train_data = 'data/your_train_data.pkl'
with open(dict_path, 'rb') as inp:
    word2id = pickle.load(inp)
    id2word = pickle.load(inp)
    tag2id = pickle.load(inp)
    id2tag = pickle.load(inp)
inp.close()

with open(train_data, 'rb') as inp:
    X = pickle.load(inp)
    Y = pickle.load(inp)
inp.close()

print(type(word2id))
print(id2word)
print(tag2id)
print(id2tag)
#print(X)
#print(len(X))
#print(len(X[0]))
print(X[0])
#print(Y)
#print(len(Y))
#print(len(Y[0]))
print(Y[0])

