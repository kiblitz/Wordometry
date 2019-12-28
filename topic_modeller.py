from nltk.tokenize import TweetTokenizer, RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from textblob import TextBlob
import re
from gensim.models import Phrases, LdaModel
from gensim.corpora import Dictionary
import pandas as pd
from pprint import pprint
import json

"""
Notes:
- Wait a few minutes to complete the preprocessing and model creation stages as those are both time intensive.

Stuff to add in to make better:
- Better dataset: currently only analyzing Trump's tweets
- Good way to filter out tokens while also preserving important characteristics of twitter messages: RegexpTokenizer, TweetTokenizer, manual function
"""

# If you can get json to work properly, then this is also a viable dataset
# with open('datasets/24/00/55.json') as json_file:
#     data = json.load(json_file)
#     print(data)
#     print("hi")

# 30,000 tweets from Trump from 5/4/09 - 12/5/16
df = pd.read_csv('datasets/2016_12_05-TrumpTwitterAll.csv')
tweets = [tweet for tweet in df.loc[:, "Tweet"]]

def preprocess_one(tweet):
    def form_sentence(tweet):
        tweet_blob = TextBlob(tweet)
        return ' '.join(tweet_blob.words)

    def proper_characters(tweet):
        tweet_list = [word for word in tweet.strip().lower().split() if re.match(r'[^a-zA-Z]*$', word)]
        return ' '.join(tweet_list)

    def rem_stopwords(tweet):
        tweet_list = [word for word in tweet.strip().lower().split() if word not in stopwords.words('english')]
        return tweet_list

    def lemmatize(tweet_list):
        lemmatizer = WordNetLemmatizer()
        return [lemmatizer.lemmatize(word) for word in tweet_list]

    # Approach 1: Manual
    # return lemmatize(rem_stopwords(proper_characters(form_sentence(tweet))))

    # Approach 2: TwitterTokenizer
    # For Twitter: includes emojis and hashtags
    # tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True)

    # Approach 3: RegexpTokenizer
    tokenizer = RegexpTokenizer(r'\w+')
    tweet_list = tokenizer.tokenize(tweet)
    tweet_list = rem_stopwords(" ".join(tweet_list))
    return lemmatize(tweet_list)


def preprocess(tweets):
    tweet_list = [preprocess_one(tweet) for tweet in tweets]

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
    # Use to create Bag-of-Words when possessing a list of tweets
    dictionary = Dictionary(tweet_list)
    print("Passed dictionary creation...")
    # Filter out words that occur less than 10 documents, or more than 50% of the documents.
    dictionary.filter_extremes(no_below=10, no_above=0.5)
    corpus = [dictionary.doc2bow(tweet) for tweet in tweet_list]

    print("Number of Unique Words:", str(len(dictionary)))
    print("Number of documents:", str(len(corpus)))
    return tweet_list, dictionary, corpus


t, d, c = preprocess(tweets)

# Create the model
temp = d[0]
id2word = d.id2token

numTopics = 20
chunkSize = 8000
passes = 10
iterations = 100

model = LdaModel(corpus=c, id2word=id2word, chunksize=chunkSize, alpha='auto', eta='auto',
                 iterations=iterations, num_topics=numTopics, passes=passes, eval_every=None)

print("Completed LDA model training...")

# Model Analysis
topics = model.top_topics(corpus=c,topn=5)
# Average topic coherence is the sum of topic coherences of all topics, divided by the number of topics.
avg_topic_coherence = sum([t[1] for t in topics]) / numTopics
print('Average topic coherence: %.4f.' % avg_topic_coherence)
pprint(topics)