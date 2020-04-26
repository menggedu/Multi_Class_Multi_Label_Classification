import tensorflow as tf
from model.transformer.layers import CustomSchedule
from utils.metrics import micro_f1,macro_f1
from model.transformer.utils import create_padding_mask
from model.transformer.trans_former_model import Transformer
import time
import os
from utils.config import  root
def cal_loss(params):
    learning_rate = CustomSchedule(params['d_model'])

    optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98,
                                         epsilon=1e-9)


def train(params, train_dataset, test_dataset):
    print("model training start")
    #导入模型
    transformer = Transformer(params['num_layers'], params['d_model'], params['num_heads'],
                              params['dff'],params['input_vocab_size'], params['output_dim'],
                              params['maximum_position_encoding'],
                              params['dropout_rate'])

    learning_rate = CustomSchedule(params['d_model'])

    optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98,
                                         epsilon=1e-9)

    loss_function = tf.keras.losses.BinaryCrossentropy(from_logits=False, reduction='none')
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.BinaryAccuracy(
        name='train_accuracy')

    checkpoint_path = "data/checkpoints/train_transformer"

    ckpt = tf.train.Checkpoint(transformer=transformer,
                               optimizer=optimizer)

    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

    # 如果检查点存在，则恢复最新的检查点。
    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print('Latest checkpoint restored!!')


    # train_step_signature = [
    #     tf.TensorSpec(shape=(None, None), dtype=tf.int32),
    #     tf.TensorSpec(shape=(None, None), dtype=tf.float32),
    # ]
    # @tf.function(input_signature=train_step_signature)
    def train_step(inp, tar):
        enc_padding_mask = create_padding_mask(inp)

        with tf.GradientTape() as tape:
            predictions = transformer(inp, True, enc_padding_mask=enc_padding_mask)
            loss = loss_function(tar, predictions)
        gradients = tape.gradient(loss, transformer.trainable_variables)
        optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))

        train_loss(loss)
        train_accuracy(tar, predictions)

        mi_f1 = micro_f1(tar, predictions)
        ma_f1 = macro_f1(tar, predictions)
        return mi_f1, ma_f1

    def predict(inp, tar, enc_padding_mask):
        predictions = transformer(inp, False, enc_padding_mask=enc_padding_mask)
        mi_f1 = micro_f1(tar, predictions)
        ma_f1 = macro_f1(tar, predictions)
        return mi_f1, ma_f1

    EPOCHS = params['EPOCHS']
    for epoch in range(EPOCHS):
        start = time.time()

        train_loss.reset_states()
        train_accuracy.reset_states()

        # inp -> portuguese, tar -> english
        for (batch, (inp, tar)) in enumerate(train_dataset):
            mic_f1, mac_f1 = train_step(inp, tar)

            if batch % 50 == 0:
                test_input, test_target = next(iter(test_dataset))
                enc_padding_mask = create_padding_mask(test_input)
                val_mic_f1, val_mac_f1 = predict(test_input, test_target, enc_padding_mask)

                print(
                    'Epoch {} Batch {} Loss {:.4f} micro_f1 {:.4f} macro_f1 {:.4f} val_micro_f1 {:.4f} val_macro_f1 {:.4f}'.format(
                        epoch + 1, batch, train_loss.result(), mic_f1, mac_f1, val_mic_f1, val_mac_f1))

        if (epoch + 1) % 5 == 0:
            ckpt_save_path = ckpt_manager.save()
            print('Saving checkpoint for epoch {} at {}'.format(epoch + 1,
                                                                ckpt_save_path))

        print('Epoch {} Loss {:.4f} Accuracy {:.4f}'.format(epoch + 1,
                                                            train_loss.result(),
                                                            train_accuracy.result()))

        print('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))

if __name__ == '__main__':
    from model.transformer.utils import get_params, get_train_params
    from utils.data_loader import build_data
    from model.transformer.utils import prepare_data

    params = get_params()
    train_params = get_train_params()

    X_train, X_test, y_train, y_test, vocab, mlb = build_data(params)
    train_dataset, test_dataset = prepare_data(params, X_train, y_train, X_test, y_test)

    train(train_params, train_dataset, test_dataset)



