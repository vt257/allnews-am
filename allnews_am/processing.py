"""Text processing methods."""
from nltk.corpus.reader import ConllCorpusReader
from nltk.util import LazyMap, LazyConcatenation
from nltk.tag import map_tag

import allnews_am.tokenizer.tokenizer as tokenizer

STANDARDIZATION = {
    ':': '։',  # Colon with Armenian վերջակետ
}


class ConllReader(ConllCorpusReader):
    """ConllCorpusReader that allows getting columns other than chunks.

    See https://www.nltk.org/_modules/nltk/corpus/reader/conll.html for more
    documentation and the source code.

    Attributes:
        root: The root folder of the dataset file(s).
        fileids: A name of a file or names of the files in the root folder to
            load.
        columntypes: The types of columns in the conll file, as a sequence.
    """

    def iob_words(self, fileids=None, tagset=None,
                  column=ConllCorpusReader.CHUNK):
        """Returns IOB annotations as tuples.

        Args:
            fileids: The list of fileids that make up this corpus.
            tagset: The tagset.
            column: The column to get the IOB annotations from, e.g. 'ne' for
                named entities or 'pos' for POS tags.

        Returns:
            A list of word/tag/IOB tuples.
        """
        self._require(self.WORDS, self.POS, self.CHUNK)

        def get_iob_words(grid):
            return self._get_iob_words(grid, tagset, column)

        return LazyConcatenation(LazyMap(get_iob_words, self._grids(fileids)))

    def iob_sents(self, fileids=None, tagset=None,
                  column=ConllCorpusReader.CHUNK):
        """Returns a list of lists of IOB annotated sentences.

        Args:
            fileids: The list of fileids that make up this corpus.
            tagset: The tagset.
            column: The column to get the IOB annotations from, e.g. 'ne' for
                named entities or 'pos' for POS tags.

        Returns:
            A list of lists of word/tag/IOB tuples.
        """
        self._require(self.WORDS, self.POS, self.CHUNK)

        def get_iob_words(grid):
            return self._get_iob_words(grid, tagset, column)

        return LazyMap(get_iob_words, self._grids(fileids))

    def _get_iob_words(self, grid, tagset=None, column=ConllCorpusReader.CHUNK):
        pos_tags = self._get_column(grid, self._colmap['pos'])
        if tagset and tagset != self._tagset:
            pos_tags = [map_tag(self._tagset, tagset, t) for t in pos_tags]
        return list(zip(self._get_column(grid, self._colmap['words']), pos_tags,
                        self._get_column(grid, self._colmap[column])))


def tokenize(s):
    """Tokenizes a string into a sequence of sentences and words.

    Args:
        s: The string to tokenize.

    Returns:
        A Sequence of sequences, where the top level sequence are sentences and
        each sentence is a sequence of string tokens.
    """
    s = s.strip()
    del_last_token = False
    if len(s) > 0 and s[-1] != ':':
        s += ':'  # Required for correct tokenization.
        del_last_token = True
    t = tokenizer.Tokenizer(s)
    t.segmentation().tokenization()
    # Segments are the sentences.
    # Tokens are the words and punctuation.
    tokenized = [
        # There care cases where 'ե՛ւ' would be represented twice, such as
        # ('1-2', 'ե՛ւ'), (1, 'եւ'), (2, '՛'). Hence, the check for integer as
        # the first element of the tuple.
        [t[1] for t in sentence['tokens'] if isinstance(t[0], int)]
        for sentence in t.segments
    ]
    if del_last_token:
        if tokenized and tokenized[-1] and tokenized[-1][-1]:
            del(tokenized[-1][-1])  # Delete the last added token ':'
    return tokenized


def standardize(sentences):
    """Standardizes the tokens.

    This includes, for instance, replacing colon '։' with Armenian ':' վերջակետ.

    Args:
        sentences: A sequence of sequences of tokens (sentence / words).

    Returns:
        A standardized sequence of sequences.
    """
    return [
        [STANDARDIZATION[t] if t in STANDARDIZATION else t for t in tokens]
        for tokens in sentences
    ]


def standardize_iob(sentences):
    """Standardizes the words in IOB sentences.

    This includes, for instance, replacing colon '։' with Armenian ':' վերջակետ.

    Args:
        sentences: A sequence of sequences of tokens as word/pos/IOB tuples.

    Returns:
        A standardized sequence of sequences.
    """
    return [
        [(STANDARDIZATION[t[0]], *t[1:]) if t[0] in STANDARDIZATION else t
         for t in sentence] for sentence in sentences
    ]
