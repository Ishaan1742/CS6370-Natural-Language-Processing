from util import *
from nltk.corpus import stopwords

class StopwordRemoval():

    def fromList(self, text):
        """
        Stopword Removal from a list of tokenized sentences

        Parameters
        ----------
        arg1 : list
            A list of lists where each sub-list is a sequence of tokens
            representing a sentence

        Returns
        -------
        list
            A list of lists where each sub-list is a sequence of tokens
            representing a sentence with stopwords removed
        """

        # Initialize a list to store the stopword removed sentences
        stopwordRemovedText = []

        # Get the list of English stopwords from NLTK
        stop_words = set(stopwords.words('english'))
        # Write the stop words to a file
        with open("stopwordsNLTK.txt", "w") as f:
            f.write("\n".join(stop_words))

        # Iterate through each sentence in the text
        for sentence in text:
            # Remove stopwords from the current sentence
            filtered_sentence = [word for word in sentence if word.lower() not in stop_words]
            # Add the filtered sentence to the list
            stopwordRemovedText.append(filtered_sentence)

        return stopwordRemovedText
    
    def bottomUpStopwordRemoval(self, text, stopwords_threshold=0.05):
        """
        Stopword Removal from a list of tokenized sentences

        Parameters
        ----------
        arg1 : list
            A list of lists where each sub-list is a sequence of tokens
            representing a sentence

        Returns
        -------
        list
            A list of lists where each sub-list is a sequence of tokens
            representing a sentence with stopwords removed
        """

        # Initialize a list to store the stopword removed sentences
        stopwordRemovedText = []

        word_freq = {}
        for sentence in text:
            for word in sentence:
                if word.lower() in word_freq:
                    word_freq[word.lower()] += 1
                else:
                    word_freq[word.lower()] = 1
        
        # Sort the word frequency dictionary in descending order
        sorted_word_freq = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        # Stop words are the most frequent words in the text
        stop_words = set([word for word, _ in sorted_word_freq[:int(len(sorted_word_freq) * stopwords_threshold)]])
        # Write the stop words to a file
        with open("stopwordsBottomUp.txt", "w") as f:
            f.write("\n".join(stop_words))
        # Iterate through each sentence in the text
        for sentence in text:
            # Remove stopwords from the current sentence
            filtered_sentence = [word for word in sentence if word.lower() not in stop_words]
            # Add the filtered sentence to the list
            stopwordRemovedText.append(filtered_sentence)

        return stopwordRemovedText
    