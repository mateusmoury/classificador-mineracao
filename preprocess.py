from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import string

class preprocess:
    def __init__(self):
        self.tokenizer = word_tokenize
        self.stemmer = PorterStemmer()
        self.punct = string.punctuation
        self.digits = string.digits
        self.stop = stopwords.words("english")

    def __process(self, snt):
        snt = self.tokenizer(snt)
        snt = [self.stemmer.stem_word(wrd) for wrd in snt if \
                    wrd not in self.stop and \
                    wrd not in self.digits and \
                    wrd not in self.punct ]
        return snt

    def process(self, snts):
        return [self.__process(snt) for snt in snts]