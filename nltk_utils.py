#import requiered librairies
import nltk
import numpy as np
#nltk.download('punkt')
from nltk.stem.porter import PorterStemmer

#initialize PorterStemmer
stemmer = PorterStemmer()

#tokenize function
def tokenize(sentence):
    """
    split sentence into array of words/tokens
    """
    return nltk.word_tokenize(sentence)

#lemmatization function
def stem(word):
    """
    to find  the root form of a word
    """
    return stemmer.stem(word, to_lowercase=True)

#function to create a bag of words[0,1] (1 for existing words else 0)
def bag_of_words(tokenized_sentence, words):
    """
    return bag of words array: 1 for each known word that exists in the sentence
    and 0 otherwise
    """
    sentence_words = [stem(w) for w in tokenized_sentence]
    bag = np.zeros(len(words), dtype=np.float32)
    for i, w in enumerate(words):
        if w in sentence_words:
            bag[i] = 1
    return bag