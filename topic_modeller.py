from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from textblob import TextBlob
import re
from gensim.models import Phrases, LdaModel
from gensim.corpora import Dictionary
import pandas as pd
import json

# with open('24/00/55.json') as json_file:
#     data = json.load(json_file)
#     print(data)
#     print("hi")

# 30,000 tweets from Trump from 5/4/09 - 12/5/16
df = pd.read_csv('2016_12_05-TrumpTwitterAll.csv')
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

    # return lemmatize(rem_stopwords(proper_characters(form_sentence(tweet))))

    # For Twitter: includes emojis and hashtags
    tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True)
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
    # Filter out words that occur less than 20 documents, or more than 50% of the documents.
    # dictionary.filter_extremes(no_below=10, no_above=0.5)
    corpus = [dictionary.doc2bow(tweet) for tweet in tweet_list]

    print("Number of Unique Words:", str(len(dictionary)))
    print("Number of documents:", str(len(corpus)))
    return tweet_list, dictionary, corpus

t, d, c = preprocess(tweets)

# print(t[:100])
# print(d)
# print(c[:5])

id2word = d.id2token
model = LdaModel(corpus=c,
                 id2word=id2word,
                 )

"""
print(preprocess_one("This is a cooool #dummysmiley: :-) :-P <3 and some arrows < > -> <--"))

t, d, c, = preprocess(["â #ireland consumer price index (mom) climbed from previous 0.2% to 0.5% in may #blog #silver #gold #forex",
                       'I was playing with my friends with whom I used to play, when you called me yesterday',
                       "Keep your phone nearby!\nThis weekend, I’ll be calling grassroots donors to thank them for chipping in to our campaign. Will you pitch in $3 tonight? I hope we’ll talk soon.",
                       "What are realistic expectations for Marshawn Lynch in his return to the Seahawks? @bcondotta & @A_Jude discuss in the latest Read Optional podcast",
                       "This is a cooool #dummysmiley: :-) :-P <3 and some arrows < > -> <--",
                       "This is a momentous achievement for the urban poor and the middle class. This initiative has been marked by transparency, use of technology and rapid implementation. I congratulate entire team at @mohua_india for their hardwork to ensure every Indian has a roof over their head."])


"""