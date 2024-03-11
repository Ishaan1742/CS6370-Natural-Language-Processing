from util import *
import nltk

class Tokenization():

    def naive(self, text):
        """
        Tokenization using a Naive Approach

        Parameters
        ----------
        arg1 : list
            A list of strings where each string is a single sentence

        Returns
        -------
        list
            A list of lists where each sub-list is a sequence of tokens
        """

        # Initialize a list to store the tokenized sentences
        tokenizedText = []

        # Iterate through each sentence in the input text
        for sentence in text:
            # Initialize a list to store tokens for the current sentence
            tokens = []
            
            # Initialize variables to keep track of token boundaries
            start = 0
            end = 0
            
            # Iterate through each character in the sentence
            for char in sentence:
                # Check if the character is a delimiter
                if char in [' ', '.', ',', ';', ':', '!', '?', '"', "'", '(', ')', '[', ']', '{', '}']:
                    # If the start and end boundaries are not the same, append the token to the list
                    if start != end:
                        tokens.append(sentence[start:end].lower())
                    # Reset the boundaries for the next token
                    start = end + 1
                # Increment the end boundary for the next character
                end += 1
            
            # Append the last token to the list
            if start != end:
                tokens.append(sentence[start:end].lower())

            # Append the list of tokens for the current sentence to the tokenizedText list
            tokenizedText.append(tokens)
        
        # Return the tokenized text
        return tokenizedText

    def pennTreeBank(self, text):
        """
        Tokenization using the Penn Tree Bank Tokenizer

        Parameters
        ----------
        arg1 : list
            A list of strings where each string is a single sentence

        Returns
        -------
        list
            A list of lists where each sub-list is a sequence of tokens
        """

        # Initialize a list to store the tokenized sentences
        tokenizedText = []

        # Create a Penn Tree Bank Tokenizer object
        tokenizer = nltk.tokenize.TreebankWordTokenizer()

        # Iterate through each sentence in the text
        for sentence in text:
            # Tokenize the sentence using Penn Tree Bank Tokenizer
            tokens = tokenizer.tokenize(sentence)
            # Add the tokenized sentence to the list
            tokenizedText.append(tokens)

        return tokenizedText