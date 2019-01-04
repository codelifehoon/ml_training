from gensim.models import word2vec

mode = word2vec.Word2Vec.load("word2vec_data.model")
result = mode.wv.most_similar(positive=["대통령","24일"])
print(result)
