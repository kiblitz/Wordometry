import pickle
from gensim.models import Word2Vec, KeyedVectors
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


download_word2vec_model('glove-twitter-25', 'saved_models/glove-twitter-25')
model_glove_twitter = retrieve_word2vec_model('saved_models/glove-twitter-25')

model_name = 'saved_models/model_glove_twitter'
model_glove_twitter.save(model_name)

model_glove_twitter = KeyedVectors.load(model_name)


def evaluate_analogies(model, fname, numLines=10000, top=10):
    correct = 0
    total = 0
    with open(fname) as analogies:
        for line in analogies.readlines()[:numLines]:
            words = line.strip().lower().split()
            if words[0] == ':':
                continue

            try:
                for pair in model.most_similar(positive=[words[1], words[2]], negative=[words[0]], topn=top):
                    if pair[0] == words[3]:
                        correct += 1
                total += 1
            except:
                continue

    print(correct, total)
    return correct/total

# print(evaluate_analogies(model_glove_twitter, "analogies_test.txt"))
score, d = model_glove_twitter.evaluate_word_analogies("datasets/analogies_test.txt")
print(score)


# print(model_glove_twitter.('england', 'crumpet'))
# print(model_glove_twitter.n_similarity(['woman', 'boy'], ['man', 'girl']))

# print(model_glove_twitter.most_similar_cosmul('sex',topn=15))
# print(model_glove_twitter.most_similar(positive = ['staircase', 'ladders'],negative = ['escalators']))

# print(model_glove_twitter.most_similar(positive = ['greece', 'baghdad'],negative = ['athens'], topn=10))

