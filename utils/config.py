import os
import pathlib


root = pathlib.Path(os.path.abspath(__file__)).parent.parent

vocab_path = os.path.join(root,'data','vocab.txt')