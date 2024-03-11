from util import *
import nltk
import re

class SentenceSegmentation():

    def naive(self, text):
        """
        Sentence Segmentation using Regular Expressions

        Parameters
        ----------
        arg1 : str
            A string (a bunch of sentences)

        Returns
        -------
        list
            A list of strings where each string is a single sentence
        """
        # Define the regex pattern for sentence segmentation i.e punctuation followed by a space
        pattern = r'(?<=[.!?:;]) +'

        # Split the text based on the regex pattern
        segmented_text = re.split(pattern, text)

        return segmented_text


    def punkt(self, text):
        """
        Sentence Segmentation using the Punkt Tokenizer

        Parameters
        ----------
        arg1 : str
            A string (a bunch of sentences)

        Returns
        -------
        list
            A list of strings where each string is a single sentence
        """

        # Initialize an empty list to store segmented sentences
        segmentedText = []

        # Use the Punkt Tokenizer from NLTK to segment the text into sentences
        tokenizer = nltk.tokenize.punkt.PunktSentenceTokenizer()
        segmentedText = tokenizer.tokenize(text)

        return segmentedText