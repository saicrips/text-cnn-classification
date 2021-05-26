import tensorflow as tf

from tensorflow.keras import layers

import numpy as numpy

def load_IMDB_dataset():
    imdb = tf.keras.datasets.imdb
    return imdb.load_data(num_words=10000)

def decode_review(text):
    imdb = tf.keras.datasets.imdb
    word_index = imdb.get_word_index()
    word_index = {k : (v+3) for k,v in word_index.items()}
    word_index["<PAD>"] = 0
    word_index["<START>"] = 1
    word_index["<UNK>"] = 2  # unknown
    word_index["<UNUSED>"] = 3

    reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

    return ' '.join([reverse_word_index.get(i, '?') for i in text])

def normalize_text_length(texts, pad_value,maxlen=256):
    output = []
    for text in texts:
        output.append(tf.keras.preprocessing.sequence.pad_sequences(text,
                                                                value=pad_value,
                                                                padding='post',
                                                                maxlen=maxlen))
    return output


class SimpleTextClass(tf.keras.Model):
    def __init__(self, vocab_size, embed_size):
        super(SimpleTextClass, self).__init__()

        self.embedding = layers.Embedding(vocab_size, embed_size)   #(batch, sequence, embedding)
        self.pool = layers.GlobalAveragePooling1D()     #sequeceの次元方向に平均値をもとめて、固定長のベクトルを返す
        self.layer = layers.Dense(embed_size, activation='relu')
        self.output_layer = layers.Dense(1, activation='sigmoid')
    
    def call(self, x, training=None):
        embed = self.embedding(x)
        pooled = self.pool(embed)
        x1 = self.layer(pooled)
        outputs = self.output_layer(x1)
        return outputs


def train(x_train,
          y_train,
          vocab_size,
          embed_size,
          epochs,
          batch_size,
          validation_data,
          test_data=None,
          test_labels=None):

    model = SimpleTextClass(vocab_size, embed_size)
    
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    history = model.fit(x_train,
                        y_train,
                        epochs=epochs,
                        batch_size=batch_size,
                        validation_data=validation_data,
                        verbose=1)

    if test_data is not None and test_labels is not None:
        results = model.evaluate(test_data, test_labels, verbose=2)
        print(results)

if __name__ =='__main__':
    (train_data, train_labels), (test_data, test_labels) = load_IMDB_dataset()
    (train_data, test_data) = normalize_text_length((train_data, test_data), 0, 256)

    vocab_size = 10000

    x_val = train_data[:10000]
    partial_x_train = train_data[10000:]

    y_val = train_labels[:10000]
    partial_y_train = train_labels[10000:]

    train(train_data, train_labels, 
          vocab_size, 16,
          40, 512, 
          (x_val, y_val),
          test_data, test_labels)
    
    
    