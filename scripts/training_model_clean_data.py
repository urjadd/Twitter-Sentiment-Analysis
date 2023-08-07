import pandas as pd

from commons.generic_constants import GenericConstants
from scripts.sentiment_analysis_model import SentimentAnalysisModel


class TrainingModelCleanData(SentimentAnalysisModel):

    def __init__(self):
        super().__init__()

    def clean_twitter_data(self):
        input_data_df = self.get_input_data()
        self.preprocessing_data(input_data_df)

    @staticmethod
    def get_input_data():
        df = pd.read_csv(GenericConstants.UNCLEANED_DATA_FILE, encoding='ISO-8859-1', header=None, engine='python')
        df.columns = GenericConstants.UNCLEANED_DATA_FILE_COLUMNS
        return df

    def preprocessing_data(self, data_df):
        data_df["Sentiment"].replace({4: 1}, inplace=True)
        data_df.Tweet = data_df.Tweet.str.lower()
        data_df = data_df.drop_duplicates(subset=['Tweet'], keep=False)
        data_df['tokens'] = data_df.Tweet.apply(lambda tweet: self.clean_tweets(tweet))
        data_df = data_df[data_df['tokens'].str.len() != 0]
        data_df = data_df[['Sentiment', 'tokens']]
        data_df.rename(columns={'tokens': 'tweet'}, inplace=True)

        negative_data_df = data_df.loc[data_df.Sentiment == 0]
        negative_data_df.reset_index(drop=True)
        positive_data_df = data_df.loc[data_df.Sentiment == 1]
        positive_data_df.reset_index(drop=True)
        training_data = pd.concat([negative_data_df[:40000], positive_data_df[:40000]])
        testing_data = pd.concat([negative_data_df[41001:43000], positive_data_df[41001:43000]])

        training_data.to_csv(GenericConstants.TRAINING_FILE, index=False)
        testing_data.to_csv(GenericConstants.TESTING_FILE, index=False)
