import tensorflow as tf
from utils.metrics import macro_f1,micro_f1
from model.transformer.trans_former_model import Transformer
from model.transformer.utils import create_padding_mask
from sklearn.metrics import f1_score
import tqdm
import numpy as np




def test(params, test_dataset):
    print("Building the model ...")
    transformer = Transformer(params['num_layers'], params['d_model'], params['num_heads'],
                              params['dff'], params['input_vocab_size'], params['output_dim'],
                              params['maximum_position_encoding'],
                              params['dropout_rate'])
    checkpoint_dir = '/home/ma-user/work/project2/data/checkpoints/train_transformer'

    print("Creating the checkpoint manager")
    checkpoint = tf.train.Checkpoint(transformer=transformer)
    checkpoint_manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=5)
    checkpoint.restore(checkpoint_manager.latest_checkpoint)
    if checkpoint_manager.latest_checkpoint:
        print("Restored from {}".format(checkpoint_manager.latest_checkpoint))
    else:
        print("Initializing from scratch.")
    print("Model restored")


    def evaluate(test_dataset):
        predictions = []
        tars = []
        for (batch, (inp, tar)) in tqdm(enumerate(test_dataset)):
            enc_padding_mask = create_padding_mask(inp)
            predict = transformer(inp, False, enc_padding_mask=enc_padding_mask)
            predictions.append(predict)
            tars.append(tar)
        predictions = tf.concat(predictions, axis=0)
        tars = tf.concat(tars, axis=0)
        mi_f1 = micro_f1(tars, predictions)
        ma_f1 = macro_f1(tars, predictions)

        predictions = np.where(predictions > 0.5, 1, 0)
        tars = np.where(tars > 0.5, 1, 0)

        smaple_f1 = f1_score(tars, predictions, average='samples')
        return mi_f1, ma_f1, smaple_f1, tars, predictions

    return evaluate(test_dataset)