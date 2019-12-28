import pickle
from gensim.models import Word2Vec, KeyedVectors
import gensim.downloader as api
import subprocess

model_gnews = KeyedVectors.load_word2vec_format("saved_models/GoogleNews-vectors-negative300.bin", binary=True, limit=200000)
print("Finished Loading")

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

print(evaluate_analogies(model_gnews, "datasets/analogies_test.txt"))
score, d = model_gnews.evaluate_word_analogies("datasets/analogies_test.txt")
print(score)