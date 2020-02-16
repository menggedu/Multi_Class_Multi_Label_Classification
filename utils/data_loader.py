import numpy as np
import pandas as pd
import argparse
import tensorflow as tf
import re
import jieba
from utils.config import root,vocab_path
import os
from sklearn.model_selection import train_test_split
from utils.multi_proc_utils import parallelize
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import MultiLabelBinarizer,LabelBinarizer

def load_stop_words(stopwords_path):
    stopwords = []
    with open(stopwords_path, 'r',encoding='utf-8') as f:
        for word in f.readlines():
            word = word.strip()
            stopwords.append(word)
    return stopwords

stopwords_path = 'data/stopwords/哈工大停用词表.txt'
stop_words = load_stop_words(stopwords_path)

def clean_sentence(line):
    '''清理sentence'''
    line = re.sub(
        "[a-zA-Z0-9]|[\s+\-\|\!\/\[\]\{\}_,.$%^*(+\"\')]+|[:：+——()?【】《》“”！，。？、~@#￥%……&*（）]+|题目", '', line)
    words = jieba.cut(line, cut_all=False)
    return words

def sentence_proc(sentence):
    cleaned_sentence = clean_sentence(sentence)
    remove_stopwords = [word for word in cleaned_sentence if word not in stop_words]
    return " ".join(remove_stopwords)

def proc(df):
    df['content'] = df['content'].apply(lambda x: sentence_proc(x))
    return df

def build_data(params):
    '''
    if os.path.exists(os.path.join(root, 'data', 'X_train.npy')):
        X_train = np.load(os.path.join(root, 'data', 'X_train.npy'))
        X_test = np.load(os.path.join(root, 'data', 'X_test.npy'))
        y_train = np.load(os.path.join(root, 'data', 'y_train.npy'))
        y_test = np.load(os.path.join(root, 'data', 'y_test.npy'))
        return X_train, X_test, y_train, y_test
    '''

    data = pd.read_csv(params['data_path'],header = None).rename(columns={0: 'label', 1: 'content'})
    processed_data = parallelize(data, proc)
    #word2index
    text_preprocesser = Tokenizer(num_words=params['vocab_size'], oov_token="<UNK>")
    text_preprocesser.fit_on_texts(processed_data['content'])
    #save vocab
    word_dict = text_preprocesser.word_index
    with open(params['vocab_path']+'vocab.txt', 'w', encoding='utf-8') as f:
        for k, v in word_dict.items():
            f.write(f'{k}\t{str(v)}\n')

    x = text_preprocesser.texts_to_sequences(processed_data['content'])
    # padding
    x = pad_sequences(x, maxlen=params['padding_size'], padding='post', truncating='post')
    # 划分标签

    if params['train_mode'] == "multi_label":
        processed_data['label'] = processed_data['label'].apply(lambda x: x.split())
        # 多标签编码
        mlb = MultiLabelBinarizer()
        y = mlb.fit_transform(processed_data['label'])
        # 数据集划分


    elif params['train_mode'] == "multi_class":
        processed_data['subject'] = processed_data['label'].apply(lambda x: x.split()[1])
        print("class category: ", set(processed_data['subject']))
        lb=LabelBinarizer()
        y = lb.fit_transform(processed_data['subject'])

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    # 保存数据
    # np.save(os.path.join(root, 'data', 'X_test.npy'), X_test)
    # np.save(os.path.join(root, 'data', 'y_train.npy'), y_train)
    # np.save(os.path.join(root, 'data', 'y_test.npy'), y_test)
    return X_train, X_test, y_train, y_test

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='This is the TextCNN test project.')
    parser.add_argument('-d', '--data_path', default='data/baidu_95.csv', type=str,
                        help='data path')
    parser.add_argument('-v', '--vocab_save_dir', default='data/', type=str,
                        help='data path')
    parser.add_argument('-vocab_size', default=50000, type=int, help='Limit vocab size.(default=50000)')
    parser.add_argument('-p', '--padding_size', default=200, type=int, help='Padding size of sentences.(default=128)')
    parser.add_argument('-train_mode', default='multi_class', type=str, help='multi-class or multi-label')
    params = parser.parse_args()

    print('Parameters:', params)


