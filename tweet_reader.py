import copy
import json
import nltk
import re
import os
import matplotlib.pyplot as plt
from collections import defaultdict
from pprint import pprint
from nltk.sentiment import SentimentIntensityAnalyzer
nltk.download('vader_lexicon')


class TweetReader:
    """This class reads all of the gathered tweets and divides them into their own subcorpora"""

    def __init__(self, data_path):
        self.tweet_schema = {
            "text": "",
            "country": "",
            "sentiment": "",
            "hashtags": []
        }
        self.path = data_path
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        self.tweets_by_country = defaultdict(list)
        self.tweets_by_keyword = {'immigrant': [], 'emigrant': [], 'migrant': [], 'refugee': [], 'asylum seeker': []}
        self.tweets = 0
        self.corpus_size = 0

    def handle_tweets(self):
        self.iterate_over_tweets()
        self.produce_output()

    def iterate_over_tweets(self):
        for file in os.listdir(self.path):
            if file.endswith('.txt'):
                filename = os.path.join(self.path, file)
                tweets_file = open(filename, "r")

                for line in tweets_file:
                    try:
                        tweet = json.loads(line)
                        self.process_tweet(tweet)
                    except json.decoder.JSONDecodeError:
                        continue

    def process_tweet(self, tweet):
        if tweet['truncated'] is True:
            tweet_text = tweet['extended_tweet']['full_text'].lower()
            hashtags = tweet['extended_tweet']['entities']['hashtags']
        elif tweet['truncated'] is False:
            tweet_text = tweet['text'].lower()
            hashtags = tweet['entities']['hashtags']
        else:
            return

        self.tweets += 1
        self.corpus_size += len(nltk.word_tokenize(tweet_text))

        try:
            if tweet['place'] is None or tweet['place']['country_code'].strip() == '':
                country_code = 'XX'
            else:
                country_code = tweet['place']['country_code']
        except KeyError:
            country_code = 'XX'

        """Remove all mentions, which act as proper nouns"""
        try:
            for mention in tweet['entities']['user_mentions']:
                username = mention['screen_name'].lower()
                tweet_text = tweet_text.replace(f'@{username}', '')
        except KeyError:
            pass

        new_tweet = copy.deepcopy(self.tweet_schema)
        new_tweet['text'] = tweet_text.replace('\n', ' ').replace('&amp;', '&')
        new_tweet['country'] = country_code
        new_tweet['sentiment'] = self.sentiment_analysis(tweet_text)

        for hashtag in hashtags:
            new_tweet['hashtags'].append(hashtag['text'].lower())

        """Multiple chosen keywords may appear in the tweet, which is why I don't use 'elif' here"""
        if 'immigrant' in tweet_text:
            self.tweets_by_keyword['immigrant'].append(new_tweet)
        if 'emigrant' in tweet_text:
            self.tweets_by_keyword['emigrant'].append(new_tweet)
        if re.search(r'\bmigrant', tweet_text) is not None:
            self.tweets_by_keyword['migrant'].append(new_tweet)
        if 'refugee' in tweet_text:
            self.tweets_by_keyword['refugee'].append(new_tweet)
        if 'asylum seeker' in tweet_text:
            self.tweets_by_keyword['asylum seeker'].append(new_tweet)

        self.tweets_by_country[country_code].append(new_tweet)
        print(self.tweets)

    def produce_output(self):
        keywords = []
        kw_count = []
        countries = []
        country_count = []

        for kw_list in self.tweets_by_keyword:
            number_of_tweets = len(self.tweets_by_keyword[kw_list])
            print(f'Tweets with "{kw_list}": {number_of_tweets}')
            outputpath = os.path.join(self.path, 'subcorpora', 'by_keyword')
            os.makedirs(outputpath, exist_ok=True)
            outname = os.path.join(outputpath, f'{kw_list}.json')
            with open(outname, 'w', encoding='utf-8') as outfile:
                json.dump(self.tweets_by_keyword[kw_list], outfile, ensure_ascii=False, indent=2)

            keywords.append(kw_list)
            kw_count.append(number_of_tweets)

        graph_data(keywords, kw_count, 'Distribution of search terms')

        other = 0
        located_tweets = []
        located_numbers = []

        for country in self.tweets_by_country:
            number_of_tweets = len(self.tweets_by_country[country])
            print(f'Tweets from "{country}": {number_of_tweets}')
            outputpath = os.path.join(self.path, 'subcorpora', 'by_country')
            os.makedirs(outputpath, exist_ok=True)
            outname = os.path.join(outputpath, f'{country}.json')
            with open(outname, 'w', encoding='utf-8') as outfile:
                json.dump(self.tweets_by_country[country], outfile, ensure_ascii=False, indent=2)

            if number_of_tweets > 200:
                countries.append(country)
                country_count.append(number_of_tweets)
                if country != 'XX':
                    located_tweets.append(country)
                    located_numbers.append(number_of_tweets)
            else:
                other += number_of_tweets

        countries.append('other')
        country_count.append(other)
        graph_data(countries, country_count, 'Distribution of collected tweets')

        located_tweets.append('other')
        located_numbers.append(other)
        graph_data(located_tweets, located_numbers, 'Distribution of geotagged tweets')

    def sentiment_analysis(self, text):
        compound = self.sentiment_analyzer.polarity_scores(text)["compound"]

        if compound >= 0.5:
            return "positive"
        elif compound <= -0.5:
            return "negative"
        else:
            return "neutral"


def graph_data(labels, numbers, name):
    figure_object, axes_object = plt.subplots()
    axes_object.pie(numbers,
                    labels=labels,
                    autopct='%1.2f',
                    startangle=90)
    axes_object.axis('equal')
    plt.title(name)
    plt.show()


if __name__ == "__main__":
    tweets_data_path = os.path.join('..', 'data')
    tr = TweetReader(tweets_data_path)
    tr.handle_tweets()
