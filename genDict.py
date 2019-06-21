import json
import jieba
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Chinese words segmentation with jieba')
    parser.add_argument('--dict', default='dict.txt', help='save path for dictionary')
    args = parser.parse_args()
    return args

def clean(name):
    if name == ' ':
        return None
    else:
        name = name.replace(' ', '')
        if name == '':
            return None
        return name

def count_words(word_list):
    res = {}
    for word in word_list:
        name = word['word']
        pos = word['pos']
        name = clean(name)
        if name is None:
            continue
        if name not in res:
            res[name] = {'pos': pos, 'num': 0}
        res[name]['num'] += 1
    return res

def read_files(file_name):
    with open(file_name, 'r', encoding='UTF-8') as load_f:
        lines = load_f.readlines()
    num = 0
    words = []
    for line in lines:
        data = json.loads(line)
        words.extend(data['postag'])
    return words

def write_files(count_res, file_name):
    with open(file_name, 'w', encoding='UTF-8') as write_f:
        for word in count_res:
            val = "%s %d %s\n" % (word, count_res[word]['num'], count_res[word]['pos'])
            write_f.writelines(val)

if __name__ == '__main__':
    args = parse_args()
    file_name = 'data/train_data.json'
    print('Reading %s...' % file_name)
    words = read_files(file_name)
    print('Done!')
    print('Counting words...')
    count_res = count_words(words)
    print('Done!')
    dict_name = args.dict
    print('Writing file to %s...' % dict_name)
    write_files(count_res, dict_name)
    print('Done!')
