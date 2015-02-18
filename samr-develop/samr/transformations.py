"""
This module implements several scikit-learn compatible transformers, see
scikit-learn documentation for the convension fit/transform convensions.
"""

import numpy
import re

from sklearn.linear_model import SGDClassifier
from sklearn.multiclass import fit_ovo
import nltk
from nltk import pos_tag
from nltk.tokenize import word_tokenize
from nltk.chunk import RegexpParser

from textblob import TextBlob

from corpus import importCSV


class StatelessTransform:
    """
    Base class for all transformations that do not depend on training (ie, are
    stateless).
    """
    def fit(self, X, y=None):
        return self


class ExtractText(StatelessTransform):
    """
    This should be the first transformation on a samr-develop pipeline, it extracts
    the phrase text from the richer `Datapoint` class.
    """
    def __init__(self, lowercase=False):
        self.lowercase = lowercase

    def transform(self, X):
        """
        `X` is expected to be a list of `Datapoint` instances.
        Return value is a list of `str` instances in which words were tokenized
        and are separated by a single space " ". Optionally words are also
        lowercased depending on the argument given at __init__.
        """
        print "Extracting Text"
        it = (" ".join(nltk.word_tokenize(datapoint.phrase)) for datapoint in X)
        if self.lowercase:
            return [x.lower() for x in it]
        return list(it)


class ReplaceText(StatelessTransform):
    def __init__(self, replacements):
        """
        Replacements should be a list of `(from, to)` tuples of strings.
        """
        self.rdict = dict(replacements)
        self.pat = re.compile("|".join(re.escape(origin) for origin, _ in replacements))

    def transform(self, X):
        """
        `X` is expected to be a list of `str` instances.
        Return value is also a list of `str` instances with the replacements
        applied.
        """
        print "Replacing Text"

        if not self.rdict:
            return X
        return [self.pat.sub(self._repl_fun, x) for x in X]

    def _repl_fun(self, match):
        return self.rdict[match.group()]


class MapToSynsets(StatelessTransform):
    """
    This transformation replaces words in the input with their Wordnet
    synsets[0].
    The intuition behind it is that phrases represented by synset vectors
    should be "closer" to one another (not suffer the curse of dimensionality)
    than the sparser (often poetical) words used for the reviews.

    [0] For example "bank": http://wordnetweb.princeton.edu/perl/webwn?s=bank
    """
    def transform(self, X):
        """
        `X` is expected to be a list of `str` instances.
        It returns a list of `str` instances such that the i-th element
        containins the names of the synsets of all the words in `X[i]`,
        excluding noun synsets.
        `X[i]` is internally tokenized using `str.split`, so it should be
        formatted accordingly.
        """
        print "Mapping to Synset"
        returnVal = [self._text_to_synsets(x) for x in X]

        return returnVal

    def _text_to_synsets(self, text):
        result = []
        for word in text.split():
            ss = nltk.wordnet.wordnet.synsets(word)
            result.extend(str(s) for s in ss if ".n." not in str(s))

        return " ".join(result)

class POSTagger(StatelessTransform):
    def transform(self, X):
        """
        `X` is expected to be a list of `str` instances.
        It returns a list of `str` instances such that the i-th element
        contains a list of parts of speech for that sentence, as tagged by nltk.tag.pos_tag

        `X[i]` is internally tokenized using nltk.tokenize.word_tokenizer.
        """
        print "Tagging POS"
        returnVal = [self._tag_sentences(x) for x in X]
        return returnVal

    def _tag_sentences(self, text):

        word_tuples = pos_tag(word_tokenize(text))

        grammar = """
            NP: {<DT>? <JJ>* <NN>* | <PRP>?<JJ.*>*<NN.*>+}
            P: {<IN>}
            V: {<V.* | VB.*>}
            PP: {<P> <NP>}
            VP: {<V> <NP|PP>*}
            CP:   {<JJR|JJS>}
            THAN: {<IN>}
            COMP: {<DT>?<NP><RB>?<VERB><DT>?<CP><THAN><DT>?<NP>}
            """
        chunker = RegexpParser(grammar)
        chunked_text = str(chunker.parse(word_tuples))
        cleaned_chunked_text = self._clean_chunked_text(chunked_text)

        return cleaned_chunked_text

    def _clean_chunked_text(self, chunked_text):
        chunked_text = re.sub(r"\n", '', chunked_text)
        chunked_text = re.sub(r" '?\w+/", ' ', chunked_text)
        return chunked_text

class SentimentChangerTagger(StatelessTransform):
    def transform(self, X):
        print "Tagging SentimentChangers"
        negators = importCSV('negators.csv')
        intensifiers = importCSV('intensifiers.csv')
        diminishers = importCSV('diminishers.csv')
        contrasters = importCSV('contrasters.csv')

        changers = {'negators': negators, 'intensifiers': intensifiers, 'diminishers': diminishers, 'contrasters': contrasters}
        result = [self._sentiment_tag(phrase, changers) for phrase in X]
        return result

    def _sentiment_tag(self, phrase, changers):
        results = []
        for type in changers:
            results.append(self._sub_sentiment_tag(phrase, changers[type], type))
        return " ".join(results).strip()


    def _sub_sentiment_tag(self, phrase, sentiment_list, type):
        results = [type for x in sentiment_list if x.strip().lower() in phrase.lower()]
        return " ".join(results)

class Polarity_And_Subjectivity(StatelessTransform):
    def transform(self, X):
        print "Adding polarity and subjectivity"
        returnVal = [self._add_polarity_and_subjectivity(x) for x in X]
        return returnVal

    def _add_polarity_and_subjectivity(self, phrase):
        blob = TextBlob(phrase)
        if len(blob.sentences) == 0:
            polarity = 2
            subjectivity = 0.5
        else:
            polarity = blob.sentences[0].sentiment.polarity + 2 # to prevent negative polarity for Naive Bayes
            subjectivity = blob.sentences[0].sentiment.subjectivity
        return [polarity, subjectivity]

class Densifier(StatelessTransform):
    """
    A transformation that densifies an scipy sparse matrix into a numpy ndarray
    """
    def transform(self, X, y=None):
        """
        `X` is expected to be a scipy sparse matrix.
        It returns `X` in a (dense) numpy ndarray.
        """
        return X.todense()


class ClassifierOvOAsFeatures:
    """
    A transformation that essentially implement a form of dimensionality
    reduction.
    This class uses a fast SGDClassifier configured like a linear SVM to produce
    a vector of decision functions separating target classes in a
    one-versus-rest fashion.
    It's useful to reduce the dimension bag-of-words feature-set into features
    that are richer in information.
    """
    def fit(self, X, y):
        """
        `X` is expected to be an array-like or a sparse matrix.
        `y` is expected to be an array-like containing the classes to learn.
        """
        self.classifiers = fit_ovo(SGDClassifier(), X, numpy.array(y), n_jobs=-1)[0]
        return self

    def transform(self, X, y=None):
        """
        `X` is expected to be an array-like or a sparse matrix.
        It returns a dense matrix of shape (n_samples, m_features) where
            m_features = (n_classes * (n_classes - 1)) / 2
        """
        xs = [clf.decision_function(X).reshape(-1, 1) for clf in self.classifiers]
        return numpy.hstack(xs)
