import re

from keras.models import Model
from keras.layers import Input, Embedding, LSTM, Dense, Activation, Dropout
from nltk import WordNetLemmatizer, PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

from commons.non_english_words import NonEnglishWords


class SentimentAnalysisModel:
    def __init__(self):
        pass

    def clean_tweets(self, text):
        print('-------------------------------------------------')
        print(text)
        text = self.remove_non_english_words(text)
        text = self.remove_abbreviations(text)
        text = self.remove_urls_mentions_emails(text)
        text = self.remove_punctuations(text)
        text = self.remove_numbers(text)
        text = self.remove_stopwords(text)
        print('-------------------------------------------------')
        return text

    @staticmethod
    def remove_punctuations(text):
        words_list = text.split()
        for i in range(len(words_list)):
            words_list[i] = re.sub(r'[^a-zA-Z0-9:;\n]', '', words_list[i])
        return ' '.join(words_list).strip()

    @staticmethod
    def remove_non_english_words(text):
        for key, value in NonEnglishWords.NON_ENGLISH_WORDS.items():
            text = re.sub(rf'{key}', value, text)
        return text.strip()

    @staticmethod
    def remove_abbreviations(text):
        words = text.split()
        for i in range(len(words)):
            if words[i] in NonEnglishWords.ABBREVIATIONS.keys():
                words[i] = NonEnglishWords.ABBREVIATIONS[words[i]]
        return ' '.join(words).strip()

    @staticmethod
    def remove_urls_mentions_emails(text):
        text = re.sub(r'\S@\S', '', text)
        text = re.sub(r'^@\S+', '', text)
        text = re.sub(r'http\S+|www\.\S+', '', text)
        return text.strip()

    @staticmethod
    def remove_numbers(text):
        text = re.sub(r'[0-9]', '', text)
        return text.strip()

    @staticmethod
    def remove_stopwords(text):
        words = text.split(' ')
        return ' '.join([word for word in words if word not in set(stopwords.words('english'))])

    @staticmethod
    def get_text_tokens(text):
        return word_tokenize(text)

    @staticmethod
    def get_stemmed_tokens(tokens):
        st = PorterStemmer()
        tokens = [st.stem(word) for word in tokens]
        return tokens

    @staticmethod
    def lemmatize_tokens(tokens):
        lemmatizer = WordNetLemmatizer()
        lem_text = []
        for word in tokens:
            lem_text.append(lemmatizer.lemmatize(word))
        return lem_text

    @staticmethod
    def create_model():
        inputs = Input(name='inputs', shape=[500])
        layer = Embedding(2000, 50, input_length=500)(inputs)
        layer = LSTM(64)(layer)
        layer = Dense(256, name='FC1')(layer)
        layer = Activation('relu')(layer)
        layer = Dropout(0.5)(layer)
        layer = Dense(1, name='out_layer')(layer)
        layer = Activation('sigmoid')(layer)
        model = Model(inputs=inputs, outputs=layer)
        return model

    @staticmethod
    def remove_emoticon(text):
        emoji_pattern = re.compile("["
                                   u"\U0001F600-\U0001F64F"  # emoticons
                                   u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                   u"\U0001F680-\U0001F6FF"  # transport & map symbols
                                   u"\U0001F1E0-\U0001F1FF"
                                   u"\u2640-\u2642"
                                   u"\u2600-\u2B55"
                                   u"\u200d"
                                   u"\u23cf"
                                   u"\u23e9"
                                   u"\u231a"
                                   u"\ufe0f"  # dingbats
                                   u"\u3030"  # flags (iOS)
                                   "]+", flags=re.UNICODE)
        text = emoji_pattern.sub(r'', text)
        return text
