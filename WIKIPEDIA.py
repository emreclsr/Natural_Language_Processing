########################################
# Metin Ön İşleme ve Görselleştirme
########################################

# Problem : Wikipedia metinleri içeren veri setine metin ön işleme ve görselleştirme yapınız.

#########################################################
# Görev 1: Metin Ön İşleme İşlemlerini Gerçekleştiriniz.
#########################################################

import pandas as pd
import nltk
from textblob import Word, TextBlob
import matplotlib.pyplot as plt
from wordcloud import WordCloud
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)
pd.set_option('display.float_format', lambda x: '%.2f' % x)

df = pd.read_csv("week12/Hw12/wiki_data.csv")


# Adım 1: Metin ön işleme için clean_text adında fonksiyon oluşturunuz. Fonksiyon;
# - Büyük, küçük harf dönüşümü
# - Noktalama işaretlerini çıkarma
# - Numerik ifadeleri çıkarma işlemlerini gerçekleştirmeli


def clean_text(dataframe):
    """
    Function that cleans the text of the dataframe.
    - Converts the text to lowercase
    - Removes punctuation
    - Removes numbers
    - Replace special characters
    - Remove multiple spaces

    Parameters
    ----------
    dataframe: pandas dataframe

    Returns
    -------
    dataframe: pandas dataframe

    """
    # Normalize Case Folding
    dataframe["text"] = dataframe["text"].str.lower()
    # Remove Punctuation
    dataframe["text"] = dataframe["text"].str.replace("[^\w\s]", "", regex=True)
    # Remove Numbers
    dataframe["text"] = dataframe["text"].str.replace("\d", "", regex=True)
    # Replace special characters
    dataframe["text"] = dataframe["text"].str.replace("\n", " ")
    dataframe["text"] = dataframe["text"].str.replace("â", " ")  # Çok sık geçiyor ve anlamsız olduğunu düşündüğüm bir karakter
    # Remove multiple spaces
    dataframe["text"] = dataframe["text"].str.replace("\s+", " ", regex=True)
    return dataframe


# Adım 2: Yazdığınız fonksiyonu veri seti içerisindeki tüm metinlere uygulayınız.


df = clean_text(df)


# Adım 3: Metin içinde öznitelik çıkarımı yaparken önemli olmayan kelimeleri (ben, sen, de, da, ki, ile vs.) çıkaracak
# remove_stopwords adında fonksiyon oluşturunuz.


def remove_stopwords(dataframe):
    """
    Function that removes stopwords from the text of the dataframe.

    Parameters
    ----------
    dataframe: pandas dataframe

    Returns
    -------
    dataframe: pandas dataframe

    """
    stopwords = nltk.corpus.stopwords.words("english")
    dataframe["text"] = dataframe["text"].apply(lambda x: " ".join(x for x in x.split() if x not in stopwords))
    return dataframe


# Adım 4: Yazdığınız fonksiyonu veri seti içerisindeki tüm metinlere uygulayınız.


df = remove_stopwords(df)


# Adım 5: Metinde az geçen (1000'den az, 2000'den az gibi) kelimeleri bulunuz. Ve bu kelimeleri metin içerisinden
# çıkartınız.


def remove_rare_words(dataframe, threshold=1000):
    """
    Function that removes rare words from the text of the dataframe.

    Parameters
    ----------
    dataframe: pandas dataframe
    threshold: int

    Returns
    -------
    dataframe: pandas dataframe

    """
    temp_df = pd.Series(" ".join(dataframe["text"]).split()).value_counts()
    rare_words = temp_df[temp_df < threshold].index

    dataframe["text"] = dataframe["text"].apply(lambda x: " ".join(x for x in x.split() if x not in rare_words))

    return dataframe


df = remove_rare_words(df)


# Adım 6: Metinleri tokenize edip sonuçları gözlemleyiniz.


def tokenize(dataframe):
    """
    Function that tokenizes the text of the dataframe.
    Parameters
    ----------
    dataframe: pandas dataframe

    Returns
    -------
    dataframe: pandas dataframe

    """
    dataframe["text"] = dataframe["text"].apply(lambda x: TextBlob(x).words)
    return dataframe


df = tokenize(df)


# Adım 7: Lemmatization işlemi yapınız.


def lemmatization(dataframe):
    """
    Function that lemmatizes the text of the dataframe.
    Parameters
    ----------
    dataframe: pandas dataframe

    Returns
    -------
    dataframe: pandas dataframe

    """
    dataframe["text"] = dataframe["text"].apply(lambda x: [Word(word).lemmatize() for word in x])
    return dataframe


df = lemmatization(df)


#########################################################
# Görev 2: Veriyi Görselleştiriniz
#########################################################

# Adım 1: Metindeki terimlerin frekanslarını hesaplayınız.
# Adım 2: Bir önceki adımda bulduğunuz terim frekanslarının Barplot grafiğini oluşturunuz.


def term_frequency(dataframe, plot=False, plot_threshold=5000):
    """
    Function that calculates the term frequency of the text of the dataframe.
    Parameters
    ----------
    dataframe: pandas dataframe
    plot: bool (default: False)
        If True, the term frequency of common words will be plotted.
    plot_threshold: int (default: 5000)
        The count threshold for the words to be plotted.

    Returns
    -------
    dataframe: pandas dataframe

    """
    tf = dataframe["text"].apply(lambda x: pd.Series(x, dtype="object")).stack().value_counts()
    tf = pd.DataFrame(tf).reset_index()
    tf.columns = ["words", "tf"]

    if plot:
        plt.figure(figsize=(10, 5))
        plt_th = tf[tf["tf"] > plot_threshold]
        plt.bar(plt_th["words"], plt_th["tf"])
        plt.title("Term Frequency")
        plt.xticks(rotation="vertical")
        plt.grid(axis="y")
        plt.tight_layout()

    return tf


term_frequency(df, plot=True, plot_threshold=5000)


# Adım 3: Kelimeleri WordCloud ile görselleştiriniz.


def word_cloud(dataframe):
    """
    Function that plots a word cloud of the text of the dataframe.
    Parameters
    ----------
    dataframe: pandas dataframe

    Returns
    -------
    None

    """

    text = dataframe["text"].str.join(sep=" ")
    text = " ".join([x for x in text])

    wordcloud = WordCloud(max_font_size=50,
                          max_words=100,
                          background_color="black",
                          colormap="tab10").generate(text)
    plt.figure()
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")


word_cloud(df)


########################################################################
# Görev 3: Tüm Aşamaları Tek Bir Fonksiyon Olarak Yazınız.
########################################################################

# Adım 1: Metin ön işleme işlemlerini gerçekleştiriniz.
# Adım 2: Görselleştirme işlemlerini fonksiyona argüman olarak ekleyiniz.
# Adım 3: Fonksiyonu açıklayan docstring yazınız.


def wiki_preprocessing(dataframe, rare_threshold=1000, bar_plot=False, bar_threshold=5000, plot_wordcloud=False):
    """
    Function that preprocesses the wiki dataset and plot different graphs.
    Performs the following steps:
    1. Removes unsuitable things from the text. Such as punctuation, numbers.
    2. Removes stopwords.
    3. Removes rare words according to the given threshold.
    4. Tokenizes the text.
    5. Lemmatizations the text.
    6. Plots the bar plot of term frequency of the text.
    7. Plots the word cloud of the text.

    Parameters
    ----------
    dataframe: pandas dataframe
    rare_threshold: int (default: 1000)
        The count threshold for the words to be removed.
    bar_plot: bool (default: False)
        If True, the term frequency of common words will be plotted.
    bar_threshold: int (default: 5000)
        The count threshold for the words to be plotted.
    plot_wordcloud: bool (default: False)
        If True, a word cloud will be plotted.

    Returns
    -------
    dataframe: pandas dataframe

    """
    df = clean_text(dataframe)
    df = remove_stopwords(df)
    df = remove_rare_words(dataframe, threshold=rare_threshold)
    df = tokenize(df)
    df = lemmatization(df)

    if bar_plot:
        term_frequency(df, plot=True, plot_threshold=bar_threshold)

    if plot_wordcloud:
        word_cloud(df)

    return df


wiki_preprocessing(df, rare_threshold=1000, bar_plot=True, bar_threshold=5000, plot_wordcloud=True)
