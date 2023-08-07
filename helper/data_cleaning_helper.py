import os
import re
import sys

import pandas as pd
from nltk.corpus import stopwords, words
from nltk.stem.snowball import SnowballStemmer

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from commons.generic_constants import GenericConstants


def clean_data_file():
    df = pd.read_csv(GenericConstants.UNCLEANED_DATA_FILE, encoding='ISO-8859-1', header=None, engine='python')
    df.columns = GenericConstants.UNCLEANED_DATA_FILE_COLUMNS
    stop_words = set(stopwords.words('english'))
    stemmer = SnowballStemmer('english')
    words_list = words.words()
    print(df[:1000].shape)
    for i in range(1000):
        for ind in df.index[i * 1000: (i * 1000) + 1000]:
            row = df.iloc[ind]
            row_tweet = row['Tweet']
            row_tweet = row_tweet.split()
            print(row_tweet)
            tweets_list = []
            for word in row_tweet:
                word = re.sub('[^a-zA-Z0-9\n:;]', '', word)
                word = stemmer.stem(word)
                if word.lower() not in words_list:
                    tweets_list.append(word)
            df['Tweet Non English Words'] = ','.join(tweets_list)
        df.to_csv('data_files/data.csv')


if __name__ == '__main__':
    clean_data_file()
