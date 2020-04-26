import os
import pathlib


root = pathlib.Path(os.path.abspath(__file__)).parent.parent

vocab_path = os.path.join(root,'data','vocab_new.txt')
data_path = os.path.join(root,'data','baidu_95.csv')

