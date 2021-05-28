"""character-level cnn classification

文字単位のテキストCNNの実装
(参考: https://towardsdatascience.com/character-level-cnn-with-keras-50391c3adf33)

"""

import os
import sys

import re
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import csv

TRAIN_PATH = '../only-my-datasets/ag_news/dataset/train.csv'
TEST_PATH = '../only-my-datasets/ag_news/dataset/test.csv'

def load_news_data():
    """ag_newsの学習用テスト用データを出力する

    ag_newsの学習用テスト用データの組を出力する
    ラベルの説明 1 World, 2 Sports, 3 Business, 4 Sci/Tech

    Returns:
        ([string], [int]), ([string], [int]): (学習用データ, ラベル),  (テスト用データ, ラベル)
    """

    train_data = []
    train_label_data = []
    with open(TRAIN_PATH, 'r', encoding='utf-8') as f:
        texts = csv.reader(f, delimiter=',', quotechar='"')
        for row in texts:
            text = ""
            for s in row[1:]:
                    text = text + " " + re.sub("^\s*(.-)\s*$", "%1", s).replace("\\n", "\n")
            train_data.append(text)
            train_label_data.append(int(row[0]))

    test_data = []
    test_label_data = []
    with open(TEST_PATH, 'r', encoding='utf-8') as f:
        texts = csv.reader(f, delimiter=',', quotechar='"')
        for row in texts:
            text = ""
            for s in row[1:]:
                    text = text + " " + re.sub("^\s*(.-)\s*$", "%1", s).replace("\\n", "\n")
            test_data.append(text)
            test_label_data.append(int(row[0]))


    return (np.array(train_data), np.array(train_label_data)), (np.array(test_data), np.array(test_label_data))

def character_level_encode(texts, maxlen=1014):
    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=None, char_level=True, oov_token='UNK')
    tokenizer.fit_on_texts(texts)

    alphabet = "abcdefghijklmnopqrstuvwxyz0123456789,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}"
    char_dict = {}
    for i, char in enumerate(alphabet):
        char_dict[char] = i+1

    tokenizer.word_index = char_dict.copy()
    tokenizer.word_index[tokenizer.oov_token] = max(char_dict.values()) + 1

    sequences = tokenizer.texts_to_sequences(texts)
    pad_data = tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen, padding='post')

    return np.array(pad_data, dtype='float32')


def text_cnn(filter_size, kernel_size, pool_size):
    """text用cnnモデルの出力

    指定したfilter_size, kernel_size, pool_sizeの
    畳み込み層とプーリング層のモデルをtf.keras.Sequential()で返す

    Parameters:
        filter_size(int): 畳み込み層のフィルター数
        kernel_size(int): 畳み込み層のカーネル(ウィンドウ)サイズ
        pool_size(int): 
            プーリング層のウィンドウサイズ
            0以下のときプーリング層はなし

    Returns:
        tf.keras.Sequential: 畳み込み層-活性化関数(relu)-プーリング層のモデルを返す
    """

    model = tf.keras.Sequential()
    model.add(layers.Conv1D(filter_size, kernel_size))
    model.add(layers.Activation('relu'))
    if pool_size > 0:
        model.add(layers.AveragePooling1D((pool_size)))

    return model

class TextCnnClass(tf.keras.Model):
    """畳み込みを使ったテキスト分類モデル

    Character-level Convolutional Networks for Text Classification
    (参照：https://arxiv.org/abs/1509.01626)のモデルを実装する

    Attributes:
        embbeding (tf.keras.Layer): embedding層(入力層)
        cnn_array ([tf.keras.Layer]): 畳み込み層とプーリング層のモデル
        fc (tf.keras.Layer): 全結合層
        dropout (tf.keras.Layer): ドロップアウト層
        output (tf.keras.Layer): 出力層(2値分類)
    """

    def __init__(self, vocab_size, embed_size, num_class, conv_layers, fc_layers, drop_rate=0.5):
        """
        Args:
            vocab_size(int): 文章の語彙数
            embed_size(int): embedding層の出力ベクトルサイズ
            conv_layers([[int]])): 
                [
                [],    //１層目の[filter_size, kernel_size, pool_size]
                [],    //2層目の[filter_size, kernel_size, pool_size]
                ...,
                []]    //n層目の[filter_size, kernel_size, pool_size]
            fc_layers([int]): 
                全結合層のunits数の配列
            drop_rate(float): 
                dropout層のdropoutさせる割合
        """
        super(TextCnnClass, self).__init__()

        self.embedding = layers.Embedding(vocab_size, embed_size)   #(batch, sequence, embedding)
        self.cnn_array = []
        for filter_size, kernel_size, pool_size in conv_layers:
            self.cnn_array.append(text_cnn(filter_size, kernel_size, pool_size))
        
        self.fc = []
        self.dropout = []
        for fc_layer in fc_layers:
            self.fc.append(layers.Dense(fc_layer, activation='relu'))
            self.dropout.append(layers.Dropout(drop_rate))

        self.output_layer = layers.Dense(num_class, activation='sigmoid')
    
    def call(self, x, training=None):
        x = self.embedding(x)
        for cnn in self.cnn_array:
            x = cnn(x)

        x = layers.Flatten()(x)

        for i, _ in enumerate(self.fc):
            x = self.fc[i](x)
            x = self.dropout[i](x, training=training)

        outputs = self.output_layer(x)
        return outputs

    def model(self, shape):
        """tf.keras.Modelとして出力

        build()された状態のモデルを返す

        Args:
            shape([int, ...]): モデルの入力のshape

        Returns:
            tf.keras.Model: build()されたモデル

        Examples:
            #summary()やget_weights()が使えるようになる
        """
        x = tf.keras.Input(shape=shape)
        return tf.keras.Model(inputs=x, outputs=self.call(x))


def train(x_train,
          y_train,
          vocab_size,
          embed_size,
          num_class,
          conv_layers,
          fc_layers,
          epochs,
          batch_size,
          optimizer,
          loss,
          validation_data,
          test_data=None,
          test_labels=None,
          checkpoint_path=None):

    model = TextCnnClass(vocab_size, embed_size, num_class, conv_layers, fc_layers, 0.5)
    model.model(shape=x_train[0].shape).summary()
    
    model.compile(optimizer=optimizer,
                  loss=loss,
                  metrics=['accuracy'])

    history = model.fit(x_train,
                        y_train,
                        epochs=epochs,
                        batch_size=batch_size,
                        validation_data=validation_data,
                        verbose=1)

    if checkpoint_path is not None:
        checkpoint = tf.train.Checkpoint(model=model)
        save_path = checkpoint.save(checkpoint_path)                        

    if test_data is not None and test_labels is not None:
        results = model.evaluate(test_data, test_labels)
        print(results)


def predict(data, vocab_size, embed_size, num_class, conv_layers, fc_layers, checkpoint_path):
    model = TextCnnClass(vocab_size, embed_size, num_class, conv_layers, fc_layers, 0.5)

    model.model(shape=data[0].shape).summary()
    checkpoint = tf.train.Checkpoint(model=model)
    save_path = checkpoint.save(checkpoint_path)
    checkpoint.restore(save_path)

    print(model.predict(data))


if __name__ =='__main__':
    (train_data, train_labels), (test_data, test_labels) = load_news_data()

    maxlen = 1014

    train_data = character_level_encode(train_data, maxlen)
    test_data = character_level_encode(test_data, maxlen)
    num_class = 4

    #ラベルが1,2,3,4となっているので, 0,1,2,3にする
    train_labels = [c-1 for c in train_labels]
    test_labels = [c-1 for c in test_labels]

    #分類モデルに合わせるためにワンホットベクトルにする
    train_labels = tf.keras.utils.to_categorical(train_labels)
    test_labels = tf.keras.utils.to_categorical(test_labels)

    vocab_size = 10000

    x_val = train_data[:10000]
    partial_x_train = train_data[10000:]

    y_val = train_labels[:10000]
    partial_y_train = train_labels[10000:]

    conv_layers = [[256, 7, 3],
                   [256, 7, 3],
                   [256, 3, 0],
                   [256, 3, 0],
                   [256, 3, 0],
                   [256, 3, 3]]

    fc_layers = [1024, 1024]

    checkpoint_path = 'data/TCC/'

    epochs = 20
    batch_size = 512
    loss = 'categorical_crossentropy'
    # loss='binary_crossentropy'
    optimizer = 'adam'

    train(partial_x_train, partial_y_train, 
          vocab_size+1, 69,
          num_class,
          conv_layers,
          fc_layers,
          epochs, batch_size,
          optimizer,
          loss,
          (x_val, y_val),
          test_data, test_labels,
          checkpoint_path)
    
    predict(train_data, vocab_size, 69, num_class, conv_layers, fc_layers, checkpoint_path)
    
    