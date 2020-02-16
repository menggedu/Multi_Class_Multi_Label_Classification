import pandas as pd
import numpy as np
import os
from utils.config import root
"""
计算标签达1626个。。。
"""

def preprocess_data(df):
    """
    对某一个csv文件提取知识点，删除无关列，组成新itm
    """
    item_list = df.item.tolist()
    processed_item = []
    knowledge_list = []
    for item in item_list:
        if "知识点" not in item:
            item_str_list = item.split('\n')
            knowledge_list.append(" ")
            new_item_str = " ".join(item_str_list[1:])
            processed_item.append(new_item_str)
        else:
            item_str_list = item.split('\n')
            knowledge_list.append(item_str_list[-1].strip())
            new_item_str = " ".join(item_str_list[1:-2])
            processed_item.append(new_item_str[:-5])
    new_df = pd.DataFrame({'label': knowledge_list, 'content': processed_item})
    return new_df


# 从path中读取后再加
def build_data(data_path):
    """
    data_path 为百度题库根目录
    """
    subjectName_list = os.listdir(data_path)
    data_df = pd.DataFrame(columns=['label', 'content'])
    for subjectName in subjectName_list:
        grade_sub = ",".join(subjectName.split('_'))
        subject_data_path = os.path.join(data_path, subjectName, "origin")
        subjectName_list2 = os.listdir(subject_data_path)  # 获取校标题
        for subjectName_sm in subjectName_list2:
            data_path_sub = os.path.join(subject_data_path, subjectName_sm)
            df = pd.read_csv(data_path_sub)
            new_df = preprocess_data(df)
            kn_name = subjectName_sm[:-4]
            subName = grade_sub + "," + kn_name
            # print(subName)
            new_df['label'] = new_df['label'].apply(lambda x: " ".join((subName + x).split(',')))
            data_df = pd.concat((data_df, new_df))

    data_df.to_csv(data_path + "label_content.csv", index=None)
    return data_df

def get_multi_labels(df):
    labels = set()
    for i, v in df.iterrows():
        v_list = v['label'].split()
        labels.update(v_list)
    labels_path = os.path.join(root,'data',"labels")
    with open(labels_path, 'w') as f:
        for label in labels:
            f.write(label.strip()+'/n')

    return labels