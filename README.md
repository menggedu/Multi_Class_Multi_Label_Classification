# Multi_Class_Multi_Label_Classification
Multi-Class and Multi-Label Classification tensorflow 2.0实现
## 1.TextCNN 做文本多类别、多标签分类
micro_f1:0.79  macro_f1:0.50
## 2.Transformer Encoder 改编
micro_f1: 0.90 macro_f1:0.80
## 3. Bert 
micro_f1:0.9121405700869043 macro_f1: 0.7629736506901216

# 实验结果
|数据集|模型|类别|Acc|Micro-F1|Macro-F1|备注|
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
|Baidu|FastText|95|-|0.421|0.234|epoch 1000, ngram 5, dim 50|
|Baidu|TextCnn|95|-|0.82478|0.578|epoch 10, lr 0.005, padding 128|
|Baidu|GCN|95|-|0.906432|0.8074780|gcn|
|Baidu|Transformer|95|-|0.90403605|0.79695547|transformer|
|Baidu|BERT|21|0.7958|0.941|0.163|BERT 3 layers labels result|
|Baidu|BERT|95|0.5788|0.917|0.781|only BERT|