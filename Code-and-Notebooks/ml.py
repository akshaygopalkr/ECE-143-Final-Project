import numpy as np
import tensorflow as tf
import pandas as pd
from tensorflow import keras
import keras
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import one_hot
import re

import nltk
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.stem import PorterStemmer
import string
import pickle
import matplotlib.pyplot as plt
from matplotlib.pylab import subplots
nltk.download('stopwords')
ps = PorterStemmer()
stop_words = stopwords.words('english')

def process_text(title):
    """
    Pre processes the text of each article by removing punctuation,
    lowering the case of each letter
    @param article: A string representing the article
    """

    title = re.sub(r'[^a-zA-Z]', ' ', title).lower()  # Remove all the punctuation from the article

    # lower all the cases of each word and split the article into each word separately
    title = title.split(' ')

    # Replace each word by it's stem word and remove all the stop words and numbers from the
    # text
    title = [ps.stem(word) for word in title if word not in stop_words]

    return ' '.join(title)


def process_data(df, col):
    """
    Process the data from the column of the dataframe
    :param df: the dataframe
    :param col: the column in the dataframe
    :return: the data from the column
    """
    return df[col].apply(process_text)


def create_ml_data(cleaned_data):
    """
    Creates the training data that will be fed into
    the LSTM network
    @param cleaned_data: the cleaned data
    """

    # Tokenizing text and converting text into numbers
    vocab_size = 5000

    numbered_data = [one_hot(article, vocab_size) for article in cleaned_data]

    # Pad the numbered data so each example is the same size
    padded_numbered_data = pad_sequences(numbered_data, padding='pre', maxlen=30)

    return padded_numbered_data


def run_ml_model(training_data, training_labels):
    """
    Creates the ml model that will be used
    to detect whether an article is fake or not
    """
    model = keras.Sequential()
    model.add(keras.layers.Embedding(5000, 50, input_length=30))
    model.add(keras.layers.LSTM(256))
    model.add(keras.layers.Dropout(.3))
    model.add(keras.layers.Dense(64, activation='relu'))
    model.add(keras.layers.Dropout(.3))
    model.add(keras.layers.Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    train_model = model.fit(training_data, training_labels, validation_split=0.33, epochs=25, batch_size=64)
    return train_model


def plot_model(train_model):

    # Graph the training and validation accuracy over each epoch
    # for the ML Model that uses title and authors as parameters
    fig, ax = plt.subplots()
    ax.plot(train_model.history['accuracy'])
    ax.plot(train_model.history['val_accuracy'])
    fig.suptitle('Accuracy using Title Only')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    ax.set_yticks([0.9,0.92, 0.94, 0.96, 0.98, 1])
    ax.legend(['Training data', 'Validation data'], loc='lower right')
    plt.show()


def predict_news_article(model, input_title):
    """
    Takes an input title name of an article
    and uses the ML algorithm to determine whether
    the article is fake or not based on the title
    @param model: The ML model used for predictions
    @param input_title: The title of the article
    @return: Whether the article is fake or not
    """
    # Preprocess the words from the title
    input_title = process_text(input_title)

    # Get the one-hot encoding of the title string
    numbered_data = one_hot(input_title, 5000)

    # Make the size of the input 30
    while len(numbered_data) != 30:

        if len(numbered_data) > 30:
            numbered_data.pop()

        else:
            numbered_data.insert(0, 0)

    # Use ML model to determine whether title is fake or not
    title_prediction = (model.predict([numbered_data]) > 0.5).astype("int32")

    return "fake" if title_prediction == 1 else "real"

if __name__ == '__main__:':
    nltk.download('stopwords')
    ps = PorterStemmer()
    stop_words = stopwords.words('english')
