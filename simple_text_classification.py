"""テキストの2値分類

IMDBの２値分類を行う
(参考: https://www.tensorflow.org/tutorials/keras/text_classification?hl=ja)

"""

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
        self.dropout = layers.Dropout(0.9)
        self.layer = layers.Dense(embed_size, activation='relu')
        self.output_layer = layers.Dense(1, activation='sigmoid')
    
    def call(self, x, training=None):
        x = self.embedding(x)
        x = self.pool(x)
        x = self.dropout(x, training=training)
        x = self.layer(x)
        outputs = self.output_layer(x)
        return outputs

    def model(self, shape):
        x = tf.keras.Input(shape=shape)
        return tf.keras.Model(inputs=x, outputs=self.call(x))

def train(x_train,
          y_train,
          vocab_size,
          embed_size,
          epochs,
          batch_size,
          validation_data,
          test_data=None,
          test_labels=None,
          checkpoint_path=None):

    model = SimpleTextClass(vocab_size, embed_size)
    
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    model.model(shape=x_train[0].shape).summary()

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
        results = model.evaluate(test_data, test_labels, verbose=2)
        print(results)

def predict(data, vocab_size, embed_size, checkpoint_path):
    model = SimpleTextClass(vocab_size, embed_size)

    model.model(shape=data[0].shape).summary()
    checkpoint = tf.train.Checkpoint(model=model)
    save_path = checkpoint.save(checkpoint_path)
    checkpoint.restore(save_path)

    print(model.predict(data).T)


if __name__ =='__main__':
    (train_data, train_labels), (test_data, test_labels) = load_IMDB_dataset()
    (train_data, test_data) = normalize_text_length((train_data, test_data), 0, 256)

    vocab_size = 10000

    x_val = train_data[:10000]
    partial_x_train = train_data[10000:]

    y_val = train_labels[:10000]
    partial_y_train = train_labels[10000:]

    checkpoint_path = 'data/stc_checkpoint'

    train(train_data, train_labels, 
          vocab_size, 16,
          40, 512, 
          (x_val, y_val),
          test_data, test_labels,
          checkpoint_path)
    
    predict(train_data, vocab_size, 16, checkpoint_path)
    