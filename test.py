import json
import jieba
from Metric import CWSMetric
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Chinese words segmentation with jieba')
    parser.add_argument('--test_data', default='data//test1_data_postag.json', help='test data file')
    parser.add_argument('--dict', default=None, help='dictionary path, if not set, then use original dictionary of jieba')
    parser.add_argument('--HMM', action='store_true', help='use HMM for unknown words')
    args = parser.parse_args()
    return args

def read_files(file_name):
    with open(file_name, 'r', encoding='UTF-8') as load_f:
        lines = load_f.readlines()
    num = 0
    sentances = []
    words = []
    for line in lines:
        data = json.loads(line)
        word = [tmp['word'] for tmp in data['postag']]
        if len(word) == 0:
            continue
        words.append(word)
        sentances.append(data['text'])
    return sentances, words

def cuttest(test_sent, HMM=False):
    result = jieba.cut(test_sent, HMM=HMM)
    return result

def testcase(sentances, words, HMM=False):
    metric = CWSMetric('P', 'R', 'F1')
    for sentance, word in zip(sentances, words):
        res = list(cuttest(sentance, HMM))
        metric.update(res, word)
    return metric.get()

if __name__ == '__main__':
    args = parse_args()
    sentances, words = read_files(args.test_data)
    if args.dict is not None:
        jieba.set_dictionary(args.dict)
    metrics = testcase(sentances, words, args.HMM)
    for metric in metrics:
        print('%s: %.4f' % (metric[0], metric[1]))
