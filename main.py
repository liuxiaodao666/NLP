#! -*- coding:utf-8 -*-
import json
import numpy as np
from tqdm import tqdm
import time
import logging
from sklearn.model_selection import StratifiedKFold
from keras_bert import load_trained_model_from_checkpoint, Tokenizer
from keras.optimizers import Adam
import tensorflow as tf
import os
import pandas as pd
import re
import jieba
from keras.utils.np_utils import to_categorical
from sklearn.metrics import mean_absolute_error, accuracy_score, f1_score


import pandas as pd
from keras_bert import Tokenizer
from keras_bert import load_trained_model_from_checkpoint
from keras.optimizers import Adam
from keras.layers import Dense
from keras.models import Model

# 读取训练数据
train_df = pd.read_csv('train.txt', sep='\t', header=None)
train_texts = train_df[0].tolist()
train_labels = train_df[1].tolist()

# 读取测试数据
test_df = pd.read_csv('test.txt', sep='\t', header=None)
test_texts = test_df[0].tolist()
test_labels = test_df[1].tolist()

# 构建标签映射
label_map = {'culture': 0, 'entertainment': 1, 'sports': 2, 'finance': 3, 'house': 4, 'car': 5, 'edu': 6, 'tech': 7,
             'military': 8, 'travel': 9, 'world': 10, 'agriculture': 11, 'game': 12, 'stock': 13, 'story': 14,
             'agriculture': 15}

# 加载预训练模型和配置
config_path = './bert1/chinese_L-12_H-768_A-12/bert_config.json'
checkpoint_path = './bert1/chinese_L-12_H-768_A-12/bert_model.ckpt'
dict_path = './bert1/chinese_L-12_H-768_A-12/vocab.txt'

# 加载Bert模型和Tokenizer
model = load_trained_model_from_checkpoint(config_path, checkpoint_path)
tokenizer = Tokenizer(dict_path)  # 替换为词汇表文件的路径

# 准备训练数据
train_tokens = []
train_segments = []

for text in train_texts:
    tokens, segments = tokenizer.encode(text, max_len=128)
    train_tokens.append(tokens)
    train_segments.append(segments)

# 准备测试数据
test_tokens = []
test_segments = []

for text in test_texts:
    tokens, segments = tokenizer.encode(text, max_len=128)
    test_tokens.append(tokens)
    test_segments.append(segments)

# 构建分类模型
output = Dense(units=len(label_map), activation='softmax')(model.output)
classification_model = Model(model.input, output)
classification_model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(lr=1e-5), metrics=['accuracy'])

# 模型训练
classification_model.fit([train_tokens, train_segments], train_labels, batch_size=16, epochs=5)

# 模型评估
_, accuracy = classification_model.evaluate([test_tokens, test_segments], test_labels)
print('Test Accuracy:', accuracy)




