import pandas as pd
import numpy as np
import nltk
import re
from nltk.corpus import stopwords
stop_words = list(stopwords.words('english'))
stop_words.append('one')
stop_words.append('said')
stop_words = set(stop_words)

def remove_stop_words(articles):
    """
    Removes all the stop words from the text
    in each articles
    """
    articles = re.sub(r'[^a-zA-Z]', ' ', articles)
    articles = articles.split(' ')

    return ' '.join([word for word in articles if word not in stop_words and len(word) > 1])


def word_cloud(text):
    '''
    This function create a wordcould from the text
    Input: the text string
    Output: a wordcloud image
    '''

    import numpy as np
    import pandas as pd
    from os import path
    from PIL import Image
    from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
    import matplotlib.pyplot as plt

    # In case the user needs to merge different texts together
    # text = " ".join(review for review in df.description)

    wordcloud = WordCloud(background_color='white').generate(text)
    plt.figure(figsize=(16, 8))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()
