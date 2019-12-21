import pickle
from gensim.models.word2vec import Word2Vec
import gensim.downloader as api
import subprocess

def download_word2vec_model(model_name, filename):
    filehandler = open(filename, 'wb')
    model = api.load(model_name)
    pickle.dump(model, filehandler)

def retrieve_word2vec_model(filename):
    filehandler = open(filename, 'rb')
    model = pickle.load(filehandler)
    return model

#def

def main():
    p = subprocess.Popen(["java", "Wordometry"], stdin=subprocess.PIPE)
    p.stdin.write("First line\r\n")
    p.stdin.write("Second line\r\n")
    p.stdin.write("x\r\n") # this line will not be printed into the file


download_word2vec_model('glove-twitter-25', 'glove-twitter-25')
model_glove_twitter = retrieve_word2vec_model('glove-twitter-25')

#print(model_glove_twitter.('england', 'crumpet'))
#print(model_glove_twitter.n_similarity(['woman', 'boy'], ['man', 'girl']))

print(model_glove_twitter.most_similar_cosmul('sex',topn=15))
#print(model_glove_twitter.most_similar(positive = ['staircase', 'ladders'],negative = ['escalators']))

#print()

