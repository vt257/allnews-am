"""Text processing methods."""
import allnews_am.tokenizer.tokenizer as tokenizer

STANDARDIZATION = {
    ':': '։',  # Colon with Armenian վերջակետ
}


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
