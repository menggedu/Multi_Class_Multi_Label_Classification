import argparse
import os
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras.models import load_model
from pprint import pprint
import time

from tensorflow.keras.callbacks import EarlyStopping

from utils.data_loader import build_data
from model.TextCNN.TextCNN import TextCNN
from utils.metrics import micro_f1, macro_f1


def test(model, x_test, y_true):

    print("Test...")
    y_pred = model.predict(x_test)

    y_true = tf.constant(y_true, tf.float32)
    y_pred = tf.constant(y_pred, tf.float32)
    print(micro_f1(y_true, y_pred))
    print(macro_f1(y_true, y_pred))
