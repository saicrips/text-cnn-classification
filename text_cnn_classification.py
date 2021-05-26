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
        #model.add(layers.GlobalAveragePooling1D())
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
    def __init__(self, vocab_size, embed_size, conv_layers, fc_layers, drop_rate=0.5):
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

        self.output_layer = layers.Dense(1, activation='sigmoid')
    
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
        x = tf.keras.Input(shape=shape)
        return tf.keras.Model(inputs=x, outputs=self.call(x))


def train(x_train,
          y_train,
          vocab_size,
          embed_size,
          conv_layers,
          fc_layers,
          epochs,
          batch_size,
          validation_data,
          test_data=None,
          test_labels=None,
          checkpoint_path=None):

    model = TextCnnClass(vocab_size, embed_size, conv_layers, fc_layers, 0.5)
    model.model(shape=x_train[0].shape).summary()
    
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
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
        results = model.evaluate(test_data, test_labels, verbose=2)
        print(results)


def predict(data, vocab_size, embed_size, conv_layers, fc_layers, checkpoint_path):
    model = TextCnnClass(vocab_size, embed_size, conv_layers, fc_layers, 0.5)

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

    conv_layers = [[256, 7, 3],
                [256, 7, 3],
                [256, 3, 0],
                [256, 3, 0],
                [256, 3, 0],
                [256, 3, 3]]

    fc_layers = [1024, 1024]

    checkpoint_path = 'data/TCC/'

    train(train_data, train_labels, 
          vocab_size, 69,
          conv_layers,
          fc_layers,
          20, 512, 
          (x_val, y_val),
          test_data, test_labels,
          checkpoint_path)
    
    predict(train_data, vocab_size, 69, conv_layers, fc_layers, checkpoint_path)
    
    