import pickle
from gensim.models.word2vec import Word2Vec
import gensim.downloader as api

def download_word2vec_model(model, filename):
    filehandler = open(filename, 'wb')
    model = api.load(database)
    pickle.dump(model, filehandler)

def retrieve_word2vec_model(filename):
    filehandler = open(filename, 'rb')
    model = pickle.load(filehandler)
    return model

model_glove_twitter = retrieve_word2vec_model('glove-twitter-25')
print(model_glove_twitter.most_similar('hot',topn=15))