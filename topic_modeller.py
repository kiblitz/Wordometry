from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from textblob import TextBlob
import re
from gensim.models import Phrases
from gensim.corpora import Dictionary


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

    # return lemmatize(rem_stopwords(proper_characters(form_sentence(tweet))))

    # For Twitter: includes emojis and hashtags
    tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True)
    tweet_list = tokenizer.tokenize(tweet)
    tweet_list = rem_stopwords(" ".join(tweet_list))
    return lemmatize(tweet_list)

def preprocess(tweets):
    tweet_list = [preprocess_one(tweet) for tweet in tweets]

    # Train bigrams/trigrams model only when there is a list of many tweets
    def n_grams(tweets):
        ngram = Phrases(tweets)
        for ind in range(len(tweets)):
            for word in ngram[tweets[ind]]:
                if '_' in word:
                    tweets[ind].append(word)

        return ngram

    tweet_list = n_grams(tweet_list)
    # Use to create Bag-of-Words when possessing a list of tweets
    dictionary = Dictionary(tweet_list)
    # Filter out words that occur less than 20 documents, or more than 50% of the documents.
    dictionary.filter_extremes(no_below=10, no_above=0.5)
    corpus = [dictionary.doc2bow(tweet) for tweet in tweet_list]
    return tweet_list, dictionary, corpus



print(preprocess_one("This is a cooool #dummysmiley: :-) :-P <3 and some arrows < > -> <--"))