import numpy as np
import pandas as pd
from textblob import TextBlob
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.sentiment import *
from gensim.models import Phrases
from gensim.corpora import Dictionary
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

import tensorflow as tf
from tensorflow import keras


# Approach 1: Use TextBlob # Fast but not very flexible to do it how you might want it
# 30,000 tweets from Trump from 5/4/09 - 12/5/16

def builtin(fname='datasets/2016_12_05-TrumpTwitterAll.csv'):
    df = pd.read_csv(fname)
    tweets = [tweet for tweet in df.loc[:, "Tweet"]]

    print(tweets[0])
    print(TextBlob(tweets[0]).sentiment)

    total = 0
    positive = 0
    subj = 0
    for tweet in tweets:
        if TextBlob(tweet).sentiment.polarity > 0:
            positive += 1
        subj += TextBlob(tweet).sentiment.subjectivity
        total += 1

    print(positive / total, subj / total)

builtin()

# Approach 2: Training a deep Neural Net Model using doc2vec and labelled data
# This implementation has 70% accuracy on test data
# Main limitation is the training data set: currently it is labelled as a binary variable (either 0 or 4) and some of the labelling is sketch (in my opinion)

def labelled(fname='datasets/trainingandtestdata/training.1600000.processed.noemoticon.csv', sample=20000):
    labelled_large = pd.read_csv(fname, encoding = "ISO-8859-1")

    SAMPLE_SIZE = sample
    tweets = []
    labelled_large = labelled_large.iloc[np.random.permutation(len(labelled_large))]
    labelled_large.reset_index(inplace=True, drop=True)
    for tweet in labelled_large.loc[:SAMPLE_SIZE, "Text"]:
        tweets.append(tweet)

    print("Downloaded data...")

    def preprocess(tweets):
        tokenizer = TweetTokenizer(preserve_case=False)
        tweet_list = [tokenizer.tokenize(tweet) for tweet in tweets]
        lemmatizer = WordNetLemmatizer()
        tweet_list = [[lemmatizer.lemmatize(word) for word in tweet if word not in stopwords.words('english')] for tweet in tweet_list]

        print("Passed initial Processing...")

        # Train bigrams/trigrams model only when there is a list of many tweets
        def n_grams(tweets):
            ngram = Phrases(tweets)
            for ind in range(len(tweets)):
                for word in ngram[tweets[ind]]:
                    if '_' in word:
                        tweets[ind].append(word)

            return tweets

        tweet_list = n_grams(tweet_list)
        print("Passed ngram Processing...")

        def tag_tweets(tweet_list):
            for i, tweet in enumerate(tweet_list):
                yield TaggedDocument(tweet, [i])

        tagged_tweets = list(tag_tweets(tweet_list))
        doc2vec_model = Doc2Vec(vector_size=50, min_count=2, epochs=40)
        doc2vec_model.build_vocab(tagged_tweets)
        print("Passed dictionary creation...")

        doc2vec_model.train(tagged_tweets, total_examples=doc2vec_model.corpus_count, epochs=doc2vec_model.epochs)
        print("Completed Training Doc2Vec model")
        corpus = [doc2vec_model.infer_vector(tweet) for tweet in tweet_list]

        return tweet_list, np.array(corpus)

    tweet_list, corpus = preprocess(tweets)

    labels = labelled_large.loc[:SAMPLE_SIZE, "Sentiment"].to_numpy() / 4

    split = int(SAMPLE_SIZE * 0.8)
    train_docs, test_docs = corpus[:split], corpus[split:]
    train_labels, test_labels = labels[:split], labels[split:]

    model = keras.Sequential([
        keras.layers.Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(2, activation='sigmoid')
    ])

    model.compile(optimizer=tf.keras.optimizers.SGD(lr=0.01, momentum=0.9),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    # Training model on training dataset
    model.fit(train_docs, train_labels, validation_split=0.25, epochs=10)

    # Evaluating model on testing dataset
    test_loss, test_acc = model.evaluate(test_docs, test_labels, verbose=2)

    print("Test Accuracy:", test_acc)

labelled(sample=100000)

# Approach 3: Use VADER database that is integrated into NLTK's sentiment analyzer
# Although the built-in method does all the work for you, I find this approach to be the most intuitive and accurate
def vader(fname='datasets/2016_12_05-TrumpTwitterAll.csv'):
    df = pd.read_csv(fname)
    tweets = [tweet for tweet in df.loc[:, "Tweet"]]
    tokenizer = TweetTokenizer(preserve_case=False)
    tweet_list = [tokenizer.tokenize(tweet) for tweet in tweets]
    tweets = []
    for tweet in tweet_list:
        tweets.append(" ".join(tweet))

    sent = SentimentIntensityAnalyzer()
    for tweet in tweets[:10]:
        scores = sent.polarity_scores(tweet)
        print(tweet)
        for val in sorted(scores):
            print("{0}: {1}, ".format(val, scores[val]))

vader()