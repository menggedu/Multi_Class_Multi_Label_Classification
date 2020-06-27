import json
import os
#import setproctitle
import traceback
from datetime import datetime
import sys
import numpy as np
import tensorflow as tf
from flask import Flask, request
sys.path.append("../../")
from utils.config import data_path, labels_path
from run_classifier import BaiduClassificationProcessor,create_model,flags,convert_single_example,InputExample
import tokenization
import modeling
import json
from sklearn.preprocessing import MultiLabelBinarizer
app = Flask(__name__)
config = flags.FLAGS
#------定义路径——————
root_path = os.path.abspath(os.path.join(os.getcwd(), "../.."))
bert_dir = os.path.join(root_path,"data","chinese_L-12_H-768_A-12")
model_dir = os.path.join(root_path,"model","bert","output","epochs6_baidu_95")
#-----加载标签————————
processor= BaiduClassificationProcessor()
labels = processor.get_label()
print(labels)
#=======用sklearn 处理多标签==========
"""  准备多标签处理工具，用于将概率转为文本标签 """

mlb = MultiLabelBinarizer()

mlb.fit([[label] for label in labels])

#=======构建计算图=========

global graph
input_ids_p, input_mask_p, label_ids_p, segment_ids_p = None, None, None, None

gpu_config = tf.ConfigProto()
gpu_config.gpu_options.allow_growth = True
sess = tf.Session(config=gpu_config)
graph = tf.get_default_graph()
with graph.as_default():
    print("going to restore checkpoint")
    #sess.run(tf.global_variables_initializer())  #max_seq_length
    input_ids_p = tf.placeholder(tf.int32, [1, config.max_seq_length], name="input_ids")
    input_mask_p = tf.placeholder(tf.int32, [1,config.max_seq_length], name="input_mask")

    bert_config = modeling.BertConfig.from_json_file(os.path.join(bert_dir, 'bert_config.json'))
# #def create_model(bert_config, is_training, input_ids, input_mask, segment_ids,
#                  labels, num_labels, use_one_hot_embeddings):

    (total_loss, per_example_loss, logits, probabilities)  = create_model(
        bert_config=bert_config, is_training=False, input_ids=input_ids_p, input_mask=input_mask_p, segment_ids=None,
        labels=None, num_labels=len(labels), use_one_hot_embeddings=False)

    saver = tf.train.Saver()
    saver.restore(sess, tf.train.latest_checkpoint(model_dir))

tokenizer = tokenization.FullTokenizer(
        vocab_file=os.path.join(bert_dir, 'vocab.txt'), do_lower_case=True)

@app.route('/class_predict_service', methods=['GET','POST'])
def class_predict_service():

    global graph
    with graph.as_default():
        result = {}
        result['code'] = 0
        try:
            sentence = request.args['text']
            result['text'] = sentence
            start = datetime.now()
            sentence = tokenizer.tokenize(sentence)
            sentence = " ".join(sentence)
            print('your input is:{}'.format(sentence))
            example = InputExample(guid=None, text_a=sentence, text_b=None)

            feature = convert_single_example(0,example,labels,config.max_seq_length,tokenizer)

            input_ids, input_mask, segment_ids, label_ids = convert(feature)

            print(input_ids)
            feed_dict = {input_ids_p: input_ids,
                         input_mask_p: input_mask}
            # run session get current feed_dict result
            pred_probabilities_result = sess.run([probabilities], feed_dict)[0]
            #print(pred_probabilities_result[0])
            #print(pred_probabilities_result)
            label_ids = np.where(pred_probabilities_result > 0.5, 1, 0)
            pred_label_result = mlb.inverse_transform(label_ids)[0]

            print(label_ids)
            #todo: 组合策略
            result['data'] = pred_label_result
            result["data2"] = convert_id2label(labels,pred_probabilities_result)
            print('time used: {} sec'.format((datetime.now() - start).total_seconds()))
            return json.dumps(result,ensure_ascii=False)
        except :
            traceback.print_exc()
            result['code'] = -1
            result['data'] = 'error'
            return json.dumps(result,ensure_ascii=False)

def convert(feature):
    '''
    转换为所用输入
    '''
    input_ids = np.reshape(feature.input_ids,(1, config.max_seq_length))
    input_mask=np.reshape(feature.input_mask,(1, config.max_seq_length))
    segment_ids=np.reshape(feature.segment_ids,(1, config.max_seq_length))
    label_ids = np.reshape(feature.label_id,(1, 95))
    return input_ids, input_mask, segment_ids, label_ids

def convert_id2label(labels, probabilities):
    results =[]
    for probability in probabilities:
        for i,pro in enumerate(probability):
            if pro>0.5:
                results.append(labels[i])
        return results
if __name__=="__main__":

    app.run(debug = True)

    #高中 生物 生物科学与社会 人工授精、试管婴儿等生殖技术 减数分裂与有丝分裂的比较 生物性污染 避孕的原理和方法,
    # 人类在正常情况下，男性产生的精子中常染色体数和性染色体种类的组合是（）A22+XYB22+XC44+XYD22+X或22+Y