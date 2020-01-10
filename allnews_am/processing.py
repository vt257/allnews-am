"""Text processing methods."""
import allnews_am.tokenizer.tokenizer as tokenizer


def tokenize(s):
    """Tokenizes a string into a sequence of sentences and words.

    Args:
        s: The string to tokenize.

    Returns:
        A Sequence of sequences, where the top level sequence are sentences and
        each sentence is a sequence of string tokens.
    """
    t = tokenizer.Tokenizer(s)
    t.segmentation().tokenization()
    # Segments are the sentences.
    # Tokens are the words and punctuation.
    return [
        # There care cases where 'ե՛ւ' would be represented twice, such as
        # ('1-2', 'ե՛ւ'), (1, 'եւ'), (2, '՛'). Hence, the check for integer as
        # the first element of the tuple.
        [t[1] for t in sentence['tokens'] if isinstance(t[0], int)]
        for sentence in t.segments
    ]
