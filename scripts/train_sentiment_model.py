import pandas as pd
from keras.optimizer_v1 import RMSprop
from keras.preprocessing import sequence

from commons.generic_constants import GenericConstants
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
from scripts.sentiment_analysis_model import SentimentAnalysisModel


class TrainSentimentModel(SentimentAnalysisModel):

    def __init__(self):
        super().__init__()

    def train_sentiment_model(self):
        training_data = self.get_training_data_set()
        print('Got Training Set')
        training_data['Lem_Data'] = training_data.tweet.apply(lambda x: self.process_data(x))
        print('Lemmatized Training Set')
        text, sentiment = training_data.Lem_Data, training_data.Sentiment
        sequence_matrix = self.get_sequence_matrix(text)
        print('Got Sequence Matrix')
        X_train, X_test, Y_train, Y_test = train_test_split(sequence_matrix, sentiment,
                                                            test_size=0.4, random_state=2)
        model = self.create_model()
        print('Model Created')
        model.compile(loss='binary_crossentropy', optimizer=RMSprop(), metrics=['accuracy'])
        history = model.fit(X_train, Y_train, batch_size=100, epochs=10)
        accuracy = model.evaluate(X_test, Y_test)
        print(accuracy)

    @staticmethod
    def get_training_data_set():
        df = pd.read_csv(GenericConstants.TRAINING_FILE)
        return df

    def process_data(self, tweet):
        tokens = self.get_text_tokens(tweet)
        stemmed_tokens = self.get_stemmed_tokens(tokens)
        lem_tokens = self.lemmatize_tokens(stemmed_tokens)
        return lem_tokens

    @staticmethod
    def get_sequence_matrix(tweets):
        tokens = Tokenizer(num_words=2000)
        tokens.fit_on_texts(tweets)
        seq = tokens.texts_to_sequences(tweets)
        seq_matrix = sequence.pad_sequences(seq, maxlen=500)
        return seq_matrix
