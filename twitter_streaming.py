import json
from pprint import pprint

from tweepy.streaming import StreamListener
from tweepy import OAuthHandler
from tweepy import Stream

access_token = "access token provided by the Twitter API"
access_token_secret = "secret access token provided by the Twitter API"
consumer_key = "consumer key provided by the Twitter API"
consumer_secret = "secret key provided by the Twitter API"


class StdOutListener(StreamListener):
    """This class is used to collect tweets"""
    def on_data(self, data):
        if accept(data):
            print(data)
            return True

    def on_error(self, status):
        print(status)


def accept(data):
    tweet = json.loads(data)
    if retweeted(tweet):
        return False
    elif not english(tweet):
        return False
    return True


def retweeted(tweet):
    """Retweets would produce duplicates and use too much memory; these are not collected"""
    try:
        if tweet["retweeted"] or 'RT' in tweet["text"][0:3]:
            return True
    except KeyError:
        pass
    return False


def english(tweet):
    if "en" in tweet["lang"]:
        return True
    return False


if __name__ == '__main__':

    listener = StdOutListener()
    auth = OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    stream = Stream(auth, listener, tweet_mode='extended')

    # Define keywords for filtering
    keywords = ['refugee', 'asylum seeker', 'immigrant', 'migrant', 'emigrant']
    stream.filter(track=keywords)
