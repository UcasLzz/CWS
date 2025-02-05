## Preparation
### download Baidu Datasets(百度网盘链接: https://pan.baidu.com/s/1guSzi13XmHYrURFSi7I1wQ 提取码：xeoi) and put as following:

-data

&emsp; -test1_data_postag.json

&emsp; -train_data.json

&emsp; -dev_data.json

### Python 3

install jieba

```
pip3 install jieba
```

## Generate new dictionary from Baidu Datasets
```
python3 genDict.py
```
save dictionary to specific path (default is dict.txt):
```
python3 genDict.py --dict your-path.txt
```
## Test jieba
test jieba with original dictionary, and you'll get P: 0.7350, R: 0.5745, F1: 0.6404: 
```
python3 test.py 
```
with HMM for unknown words, and you'll get P: 0.7594, R: 0.6464, F1: 0.6936:
```
python3 test.py --HMM
```
use dictionary generated, and you'll get P: 0.8781, R: 0.7762, F1: 0.8192:
```
python3 test.py --dict dict.txt  # be sure to generate dictionary before
```

## BiLSTM + CRF
```
python data_preprocessing.py
```
every epoch, print valid dataset P, R, F1
after all epoches, you'll get test dataset P: 0.8695, R: 0.8612, F1: 0.8654
```
python train.py
```


## Suggetsed Readings
Neural Architectures for Named Entity Recognition
https://aclweb.org/anthology/N16-1030

Bidirectional LSTM-CRF Models for Sequence Tagging
https://arxiv.org/pdf/1508.01991v1.pdf
