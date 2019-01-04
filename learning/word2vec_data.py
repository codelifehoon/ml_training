from gensim.models import  word2vec
import  codecs
from bs4 import  BeautifulSoup
from konlpy.tag import  Okt
import multiprocessing


fp  = codecs.open("word2vec_data_simple.txt","r",encoding="utf-8")
twitter = Okt()
results= []

for line in fp.readlines():
    r = []
    malist = twitter.pos(line,norm=True,stem=True)
    for (word,pumsa) in malist:
        if not pumsa in ["Josa","Eomi","Punctuation"]:
            r.append(word)
    results.append((" ".join(r)).strip())

output = ( (" ".join(results)).strip() )

with open("word2vec_data.wakati","w",encoding="utf-8") as fp:
        fp.write(output)

data = word2vec.LineSentence("word2vec_data.wakati")
config = { 'min_count': 1,
           'size': 300,
           'sg': 1,
           'batch_words': 10,
           'iter': 100,
           'workers': multiprocessing.cpu_count(), }

model = word2vec.Word2Vec(data,**config)
model.save("word2vec_data.model")



import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
import numpy as np





