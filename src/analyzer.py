from spacy.lang.en.stop_words import STOP_WORDS
from org.apache.lucene.analysis import Analyzer
from org.apache.pylucene.analysis import PythonAnalyzer
from org.apache.lucene.analysis.standard import StandardTokenizer, StandardAnalyzer
from org.apache.lucene.analysis.core import StopFilter, LowerCaseFilter
from org.apache.lucene.analysis.ngram import NGramTokenFilter
from org.apache.lucene.analysis.en import PorterStemFilter
from org.apache.lucene.analysis.shingle import ShingleFilter
from org.apache.lucene.analysis import CharArraySet
from org.apache.lucene.analysis.pattern import PatternReplaceFilter
from java.util.regex import Pattern


from org.apache.lucene.analysis.core import LowerCaseFilter, WhitespaceTokenizer
from org.apache.pylucene.analysis import PythonAnalyzer
from org.apache.lucene.analysis import Analyzer

class StemmingAnalyzer(PythonAnalyzer):
    def __init__(self, stop_words=None):
        PythonAnalyzer.__init__(self)
        if stop_words is None:
            stop_words = STOP_WORDS

        self.stop_word_set = CharArraySet(len(stop_words), True)
        for word in stop_words:
            self.stop_word_set.add(word)

    def createComponents(self, fieldName):
        # Step 1: Tokenize the text
        tokenizer = StandardTokenizer()
        
        # Step 2: Keep only letters using a PatternReplaceFilter
        tokenStream = PatternReplaceFilter(tokenizer, Pattern.compile("[^A-Za-z]+"), "", True)
        
        # # Step 3: Convert tokens to lowercase
        tokenStream = LowerCaseFilter(tokenStream)
        
        # Step 4: Remove stop words
        tokenStream = StopFilter(tokenStream, self.stop_word_set)
        
        # Step 5: Apply stemming
        tokenStream = PorterStemFilter(tokenStream)
        
        return Analyzer.TokenStreamComponents(tokenizer, tokenStream)


class WordLevelNGramAnalyzer(PythonAnalyzer):
    def __init__(self, stop_words=None, min_gram=2, max_gram=3):
        PythonAnalyzer.__init__(self)
        if stop_words is None:
            self.stop_words = StandardAnalyzer.ENGLISH_STOP_WORDS_SET
        else:
            self.stop_words = CharArraySet(stop_words)
        self.min_gram = min_gram
        self.max_gram = max_gram

    def createComponents(self, fieldName):
        # Step 1: Tokenize the text
        tokenizer = StandardTokenizer()
        
        # Step 2: Convert tokens to lowercase
        tokenStream = LowerCaseFilter(tokenizer)
        
        # Step 3: Remove stop words
        tokenStream = StopFilter(tokenStream, self.stop_words)
        
        # Step 4: Generate word-level N-grams
        tokenStream = ShingleFilter(tokenStream, self.min_gram, self.max_gram)
        
        return Analyzer.TokenStreamComponents(tokenizer, tokenStream)
    


class CharacterLevelNGramAnalyzer(Analyzer):
    def __init__(self, min_gram=2, max_gram=3, stop_words=None):
        self.min_gram = min_gram
        self.max_gram = max_gram
        if stop_words is None:
            self.stop_words = StandardAnalyzer.ENGLISH_STOP_WORDS_SET
        else:
            self.stop_words = CharArraySet(stop_words)

    def createComponents(self, fieldName):
        # Step 1: Tokenize the text
        tokenizer = StandardTokenizer()
        
        # Step 2: Convert tokens to lowercase
        tokenStream = LowerCaseFilter(tokenizer)
        
        # Step 3: Remove stop words
        tokenStream = StopFilter(tokenStream, self.stop_words)
        
        # Step 4: Generate character-level n-grams
        tokenStream = NGramTokenFilter(tokenStream, self.min_gram, self.max_gram)
        
        return Analyzer.TokenStreamComponents(tokenizer, tokenStream)



