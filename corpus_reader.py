import json
import matplotlib.pyplot as plt
import numpy as np
import nltk
import os
import csv
from collections import defaultdict
from pprint import pprint
from nltk.collocations import *
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer

plt.rcParams.update({'font.size': 8})
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('universal_tagset')


class CorpusReader:
    """Reads the subcorpora and produces the data for discussion"""

    def __init__(self, data_path):
        self.path = data_path
        self.total_tweets = defaultdict(int)
        self.stopwords = set(nltk.corpus.stopwords.words('english'))
        self.links_and_encoding = ['https', 'http', 'co']
        self.lemmatizer = WordNetLemmatizer()
        self.bigram_measures = nltk.collocations.BigramAssocMeasures()

    def handle_data(self):
        country_corpus = os.path.join(self.path, 'by_country')
        keyword_corpus = os.path.join(self.path, 'by_keyword')
        print('Analyzing country subcorpora...')
        self.analyze_corpus(country_corpus, 'country')
        print('Analyzing search term subcorpora...')
        self.analyze_corpus(keyword_corpus, 'term')
        print('Process complete!')

    def analyze_corpus(self, corpus, mode):
        large_subcorpora = ['asylum seeker', 'emigrant', 'immigrant', 'migrant', 'refugee',
                            'US', 'GB', 'CA', 'AU', 'IE']
        for file in os.listdir(corpus):
            if file.endswith('.json') and file.split('.')[0] in large_subcorpora:
                print(f'Processing {file}...')
                subcorpus = os.path.join(corpus, file)
                self.analyze(subcorpus, mode)

    def analyze(self, corpus, mode):
        """A 'corpus' here means a text file in either the keyword or country directory"""
        # Read the subcorpus, determining keywords and most common collocates
        with open(corpus, 'r', encoding='utf-8', errors='ignore') as f:
            subcorpus = json.load(f)
        name = os.path.basename(corpus)
        name = name.split('.')[0]
        subcorpus_keywords = defaultdict(int)
        subcorpus_adjectives = defaultdict(int)
        subcorpus_verbs = defaultdict(int)
        subcorpus_collocates = defaultdict(int)
        subcorpus_bigrams = defaultdict(int)
        subcorpus_hashtags = defaultdict(int)
        subcorpus_sentiment = defaultdict(int)
        clean_text = []

        for tweet in subcorpus:
            self.total_tweets['total'] += 1
            self.total_tweets[name] += 1
            tweet_text = tweet['text']

            for hashtag in tweet['hashtags']:
                subcorpus_hashtags[hashtag] += 1
                tweet_text = tweet_text.replace(f'#{hashtag}', '')

            keywords, adjectives, verbs = self.find_keywords(tweet_text, name)
            for keyword, occurrences in keywords.items():
                subcorpus_keywords[keyword] += occurrences

            for adj, occurrences in adjectives.items():
                subcorpus_adjectives[adj] += occurrences

            for verb, occurrences in verbs.items():
                subcorpus_verbs[verb] += occurrences

            if mode == 'country':
                bigrams = self.find_bigrams(tweet_text)
                for bigram, occurrences in bigrams.items():
                    subcorpus_bigrams[bigram] += occurrences
            elif mode == 'term':
                collocates = self.find_collocates(tweet_text, name)
                for collocate, occurrences in collocates.items():
                    subcorpus_collocates[collocate] += occurrences

            subcorpus_sentiment[tweet['sentiment']] += 1

            clean_text.append(tweet_text)
            print(self.total_tweets['total'])

        """Create a bar chart of the keywords"""
        keyword_counter = nltk.Counter(subcorpus_keywords)
        labels = []
        values = []
        for keyword, count in keyword_counter.most_common(20):
            labels.append(keyword)
            values.append(count)
        graph_data_bar(labels, values, f'Most common keywords in the {name} subcorpus',
                       'Occurrences', 'Keyword')

        adj_counter = nltk.Counter(subcorpus_adjectives)
        labels = []
        values = []
        for adj, count in adj_counter.most_common(20):
            labels.append(adj)
            values.append(count)
        graph_data_bar(labels, values, f'Most common adjectives in the {name} subcorpus',
                       'Occurrences', 'Keyword')

        verb_counter = nltk.Counter(subcorpus_verbs)
        labels = []
        values = []
        for verb, count in verb_counter.most_common(20):
            labels.append(verb)
            values.append(count)
        graph_data_bar(labels, values, f'Most common verbs in the {name} subcorpus',
                       'Occurrences', 'Keyword')

        if mode == 'country':
            """Create a list/csv of the most common bigrams"""
            bigram_counter = nltk.Counter(subcorpus_bigrams)
            outputpath = os.path.join('..', 'data', 'csv_files')
            os.makedirs(outputpath, exist_ok=True)
            file = os.path.join(outputpath, f'{name}_bigrams.csv')
            if os.path.exists(file):
                os.remove(file)
            print(f'\nMost common bigrams in the {name} corpus:')
            with open(file, mode='w') as bigram_list:
                writer = csv.writer(bigram_list, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                for n, (bigram, occurrences) in enumerate(bigram_counter.most_common(20)):
                    print(f'    {n + 1}. {bigram} ({occurrences} samples)')
                    writer.writerow([bigram, occurrences])
            input('\n')
        elif mode == 'term':
            """Create a bar graph of the most common collocates"""
            collocate_counter = nltk.Counter(subcorpus_collocates)
            labels = []
            values = []
            pre_labels = []
            post_labels = []
            pre_values = []
            post_values = []
            for (w1, w2), count in collocate_counter.most_common(20):
                if len(name.split()) == 2:
                    # This collocate is for 'asylum seeker'
                    if name.split()[1] == w1:
                        # this is a post-word collocate
                        labels.append(w2)
                        values.append(count)
                        post_labels.append(w2)
                        post_values.append(count)
                    elif name.split()[0] == w2:
                        # this is a pre-word collocate
                        labels.append(w1)
                        values.append(count)
                        pre_labels.append(w1)
                        pre_values.append(count)
                elif name == w1:
                    # this is a post-word collocate, as the search term is the first word in the bigram
                    labels.append(w2)
                    values.append(count)
                    post_labels.append(w2)
                    post_values.append(count)
                else:
                    # this is a pre-word collocate, as the search term is the second word in the bigram
                    labels.append(w1)
                    values.append(count)
                    pre_labels.append(w1)
                    pre_values.append(count)

            graph_data_bar(labels, values, f'Most common collocates of {name}',
                           'Occurrences', 'Collocate')
            graph_data_bar(pre_labels, pre_values, f'Most common pre-word collocates of {name}',
                           'Occurrences', 'Collocate')
            graph_data_bar(post_labels, post_values, f'Most common post-word collocates of {name}',
                           'Occurrences', 'Collocate')

        """Create a list/csv of the hashtags"""
        hashtag_counter = nltk.Counter(subcorpus_hashtags)
        print(f'\nMost common hashtags in the {name} corpus:')
        outputpath = os.path.join('..', 'data', 'csv_files')
        os.makedirs(outputpath, exist_ok=True)
        file = os.path.join(outputpath, f'{name}_hashtags.csv')
        if os.path.exists(file):
            os.remove(file)
        with open(file, mode='w') as hashtag_list:
            writer = csv.writer(hashtag_list, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            for n, (key, value) in enumerate(hashtag_counter.most_common(20)):
                print(f'    {n + 1}. {key} ({value} samples)')
                writer.writerow([key, value])
        input('\n')

        """"Create a pie chart of the sentiment distribution"""
        labels = ['positive', 'negative', 'neutral']
        values = [subcorpus_sentiment['positive'], subcorpus_sentiment['negative'], subcorpus_sentiment['neutral']]

        graph_data_pie(labels, values, f'Sentiment distribution in the {name} subcorpus')

        """Create clean text file for AntConc"""
        path = os.path.join(corpus, '..', 'clean')
        os.makedirs(path, exist_ok=True)
        outname = os.path.join(path, f'{name}.txt')
        with open(outname, 'w', encoding='utf-8') as outfile:
            for clean_tweet in clean_text:
                outfile.write(clean_tweet)

    def find_keywords(self, tweet, name):
        keywords = defaultdict(int)
        adjectives = defaultdict(int)
        verbs = defaultdict(int)
        tweet = tweet.replace('u.s.', 'US')
        countries = ['US', 'AU', 'IE', 'CA', 'GB']
        for word, pos in nltk.pos_tag(nltk.wordpunct_tokenize(tweet), tagset='universal'):
            if not (word == 'us' and pos == 'PRON'):
                word = self.lemmatizer.lemmatize(word, get_wordnet_pos(pos))

            if (word not in self.stopwords) and (word not in self.links_and_encoding) and \
                    (word.isalpha()) and \
                    ((name in countries) or
                     ((word != name) and (len(name.split()) == 1)) or
                     ((word not in name.split()) and (len(name.split()) == 2))):
                keywords[word] += 1

                if pos == 'ADJ':
                    adjectives[word] += 1
                elif pos == 'VERB':
                    verbs[word] += 1

        return keywords, adjectives, verbs

    def find_bigrams(self, tweet):
        bigrams = defaultdict(int)
        tokenized_tweet = nltk.wordpunct_tokenize(tweet)
        finder = BigramCollocationFinder.from_words(tokenized_tweet)
        finder.apply_word_filter(lambda w: w in self.stopwords or w in self.links_and_encoding or not w.isalpha())
        for bigram in finder.nbest(self.bigram_measures.pmi, 10):
            bigrams[bigram] += 1
        return bigrams

    def find_collocates(self, tweet, name):
        collocates = defaultdict(int)
        tokenized_tweet = nltk.wordpunct_tokenize(tweet)
        finder = BigramCollocationFinder.from_words(tokenized_tweet)

        finder.apply_ngram_filter(lambda w1, w2:
                                  w1 in self.stopwords or w2 in self.stopwords or
                                  w1 in self.links_and_encoding or w2 in self.links_and_encoding or
                                  (not w1.isalpha()) or (not w2.isalpha()))
        if len(name.split()) == 2:
            """This way, only words before 'asylum' or after 'seeker' are collected"""
            finder.apply_ngram_filter(lambda w1, w2: not (w2 == name.split()[0] or w1 == name.split()[1]))
        else:
            finder.apply_ngram_filter(lambda w1, w2: name not in [w1, w2])

        for collocate in finder.nbest(self.bigram_measures.pmi, 10):
            """This condition ensures that the collocate isn't for an instance of 'asylum' or 'seeker' alone"""
            if (name != 'asylum seeker') or (f"{' '.join(collocate)} seeker" in ' '.join(tokenized_tweet) or
                                             f"asylum {' '.join(collocate)}" in ' '.join(tokenized_tweet)):
                collocates[collocate] += 1

        return collocates


def graph_data_pie(labels, values, name):
    figure_object, axes_object = plt.subplots()
    colors = ['green', 'red', 'blue']
    axes_object.pie(values,
                    labels=labels,
                    colors=colors,
                    autopct='%1.2f',
                    startangle=90)
    axes_object.axis('equal')
    plt.title(name)
    plt.show()


def graph_data_bar(labels, values, title, y, x):
    plt.bar(labels,
            values,
            alpha=0.5)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.title(title)
    plt.xticks(labels)
    plt.show()


def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return 'n'


if __name__ == "__main__":
    corpora = os.path.join('..', 'data', 'subcorpora')
    cr = CorpusReader(corpora)
    cr.handle_data()
