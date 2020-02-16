
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import plot_model
import logging

class TextCNN(tf.keras.Model):
    def __init__(self, feature_size, num_classes, vocab_size,
                 embed_size, filter_sizes, num_filters,
                 train_mode,embedding_matrix=None,dropout_rate=0,regularizers_lambda = 0,model_img_path=None):
        super(TextCNN, self).__init__()
        self.feature_size = feature_size
        #self.num_classes = self.num_classes
        self.num_filters = num_filters
        self.embed_size= embed_size
        self.filter_sizes = filter_sizes
        self.num_filters = num_filters
        self.train_mode = train_mode
        self.model_img_path = model_img_path
        # 保存embedding 矩阵
        if embedding_matrix is None:
            self.embedding = keras.layers.Embedding(input_dim=vocab_size, output_dim=embed_size,
                                                    input_length=feature_size)
            #embed_initer = keras.initializers.RandomUniform(minval=-1, maxval=1)
            # self.embedding = keras.layers.Embedding(vocab_size, embed_size,
            #                                         embeddings_initializer=embed_initer,
            #                                         input_length=feature_size,
            #                                         name='embedding')
        else:
            #embedding_matrix = w2v_model.vectors
            self.embedding = tf.keras.layers.Embedding(vocab_size, embed_size, weights=[embedding_matrix],
                                                       trainable=False)
        self.dropout_layer = keras.layers.Dropout(dropout_rate, name='dropout')
        if self.train_mode =="multi_class":
            self.fc = keras.layers.Dense(num_classes, activation='softmax',
                                             kernel_initializer='glorot_normal',
                                             bias_initializer=keras.initializers.constant(0.1),
                                             kernel_regularizer=keras.regularizers.l2(regularizers_lambda),
                                             bias_regularizer=keras.regularizers.l2(regularizers_lambda),
                                             name='dense')
        if self.train_mode == "multi_label":
            self.fc = keras.layers.Dense(num_classes,activation = "sigmoid")

    def build_model(self,):
        print("train_mode is:", self.train_mode)
        inputs = keras.Input(shape=(self.feature_size,), name="input_data")
        logging.info("x_input.shape: %s" % str(inputs.shape))
        x = self.embedding(inputs)
        logging.info("x_emb.shape: %s" % str(x.shape))
        # reshape: assume only have one channel
        x = keras.layers.Reshape((self.feature_size, self.embed_size, 1), name="add_channel")(x)

        # create a convolution + maxpool layer for each filter size
        pool_outputs = []
        for index, filter_size in enumerate(self.filter_sizes):
            filter_shape = (filter_size, self.embed_size)
            conv = keras.layers.Conv2D(self.num_filters, filter_shape, strides=(1, 1), padding='valid',
                                       data_format='channels_last', activation='relu',
                                       # channel_last ="(batch, height, width, channels)"
                                       kernel_initializer='glorot_normal',
                                       bias_initializer=keras.initializers.constant(0.1),
                                       name='convolution_{:d}'.format(filter_size))(x)
            max_pool_shape = (self.feature_size - filter_size + 1, 1)
            pool = keras.layers.MaxPool2D(pool_size=max_pool_shape,
                                  strides=(1, 1), padding='valid',
                                  data_format='channels_last',
                                  name='max_pooling_{:d}'.format(filter_size))(conv)
            #shape= (batch_size, 1,1,num_filters)
            pool_outputs.append(pool)

        pool_outputs = keras.layers.concatenate(pool_outputs, axis=-1, name='concatenate')
        pool_outputs = keras.layers.Flatten(data_format='channels_last', name='flatten')(pool_outputs)
        pool_outputs = self.dropout_layer(pool_outputs)

        outputs = self.fc(pool_outputs)
        logging.info("y.shape: %s \n" % str(outputs.shape))
        model = keras.Model(inputs=inputs, outputs=outputs)
        if self.model_img_path:
            plot_model(model, to_file=self.model_img_path, show_shapes=True, show_layer_names=False)

        model.summary()
        return model
if __name__ == '__main__':
    batch_size = 16
    feature_size = 100
    w2v_model = None
    num_classes = 3
    vocab_size= 10000
    embed_size= 300
    filter_sizes = [2,3,4]
    num_filters = 20
    dropout_rate = 0.5
    regularizers_lambda = 0.1
    x_train = tf.ones((batch_size, feature_size))
    model = TextCNN(feature_size,w2v_model,num_classes,vocab_size,embed_size,filter_sizes,num_filters,
                    dropout_rate,regularizers_lambda).build_model()
    print(model.summary())
