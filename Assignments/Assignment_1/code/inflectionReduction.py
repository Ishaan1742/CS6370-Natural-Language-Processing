from util import *
import nltk
from nltk.stem import PorterStemmer, WordNetLemmatizer

class InflectionReduction:

    def reduce(self, text):
        """
        Stemming/Lemmatization

        Parameters
        ----------
        arg1 : list
            A list of lists where each sub-list a sequence of tokens
            representing a sentence

        Returns
        -------
        list
            A list of lists where each sub-list is a sequence of
            stemmed/lemmatized tokens representing a sentence
        """

        # Initialize a list to store the stemmed/lemmatized sentences
        reducedText = []

        # Initialize stemmer and lemmatizer objects
        stemmer = PorterStemmer()
        lemmatizer = WordNetLemmatizer()

        # Iterate through each sentence in the text
        for sentence in text:
            # Perform stemming or lemmatization on each token in the sentence
            stemmed_lemmatized_sentence = [stemmer.stem(word) for word in sentence] # Stemming using Porter Stemmer
            # stemmed_lemmatized_sentence = [lemmatizer.lemmatize(word) for word in sentence] # Lemmatization using WordNet Lemmatizer
			
            # Add the stemmed/lemmatized sentence to the list
            reducedText.append(stemmed_lemmatized_sentence)

        return reducedText
