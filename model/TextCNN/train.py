import argparse
import os
import sys
from tensorflow import keras
import tensorflow as tf
from pprint import pprint
import time
o_path = os.getcwd() # 返回当前工作目录

sys.path.append(o_path)
from tensorflow.keras.callbacks import EarlyStopping

#from data_helper import data_loader
from model.TextCNN.TextCNN import TextCNN
from utils.metrics import micro_f1, macro_f1


def train(X_train, X_test, y_train, y_test, params, save_path):
    print("\nTrain...")
    model = build_model(params)

    # parallel_model = keras.utils.multi_gpu_model(model, gpus=2)
    # parallel_model.compile(tf.optimizers.Adam(learning_rate=args.learning_rate), loss='binary_crossentropy',
    #                        metrics=[micro_f1, macro_f1])
    # keras.utils.plot_model(model, show_shapes=True, to_file=os.path.join(args.results_dir, timestamp, "model.pdf"))
    # y_train = tf.one_hot(y_train, args.num_classes)
    # tb_callback = keras.callbacks.TensorBoard(os.path.join(args.results_dir, timestamp, 'log/'),
    #                                           histogram_freq=0.1, write_graph=True,
    #                                           write_grads=True, write_images=True,
    #                                           embeddings_freq=0.5, update_freq='batch')

    print('Train...')
    early_stopping = EarlyStopping(monitor='val_micro_f1', patience=10, mode='max')

    history = model.fit(X_train, y_train,
                        batch_size=params['batch_size'],
                        epochs=params['epochs'],
                        workers=params['workers'],
                        use_multiprocessing=True,
                        callbacks=[early_stopping],
                        validation_data=(X_test, y_test))

    print("\nSaving model...")
    keras.models.save_model(model, save_path)
    pprint(history.history)


def build_model(params):
    if params["train_mode"] == 'multi_label':
        print("multi_label classification begin!")
        textcnn = TextCNN(feature_size=params['feature_size'], num_classes=params['num_classes'],
                          vocab_size=params['vocab_size'],
                          embed_size=params['embed_size'],
                          filter_sizes=params['filter_sizes'],
                          num_filters=params['num_filters'],
                          train_mode=params['train_mode'])
        model = textcnn.build_model()
        model.compile(tf.optimizers.Adam(learning_rate=params['learning_rate']),
                      loss='binary_crossentropy',
                      metrics=[micro_f1, macro_f1])

    elif params["train_mode"]=="multi_class":
        print("multi_class classification begin!")
        textcnn = TextCNN(feature_size=params['feature_size'], num_classes=params['num_classes'],
                          vocab_size=params['vocab_size'],
                          embed_size=params['embed_size'],
                          filter_sizes=params['filter_sizes'],
                          num_filters=params['num_filters'],
                          train_mode=params['train_mode'])
        model = textcnn.build_model()
        model.compile(tf.optimizers.Adam(learning_rate=params['learning_rate']),
                      loss='categorical_crossentropy',
                      metrics=[micro_f1, macro_f1])

    else:
        print("train mode does not exist !!!")

    #model.summary()
    return model