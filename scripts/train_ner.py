"""Train an NER model.

Bi-LSTM + Character Bi-LSTM + Softmax.

Example hyper-parameters and results:
--pretrained_emb=models/fastText_sg_100.model
30 epochs, LSTM dims 50/20, embeddings 100/10, recurrent_dropouts 0.5/0.5
SpatialDropout1D 0.3

"easy test"
[these are the values reported in the pioNER paper setting the benchmark.]
- test f1: 65.53
        precision   recall  f1-score   support

Train     0.8557    0.8139    0.8325    130719
Dev       0.7371    0.7207    0.7268     32528
Test      0.7495    0.5971    0.6553     53606

[Overfitting - more tuning can improve the scores.]

"difficult test" [dev]
[only correct if all of the tokens for multi-token named entity are recognized]
 - f1: 62.45
           precision    recall  f1-score   support

      PER     0.6722    0.6543    0.6632       865
      ORG     0.2986    0.3986    0.3414       424
      LOC     0.6790    0.6852    0.6821      1849

micro avg     0.6115    0.6380    0.6245      3138
macro avg     0.6257    0.6380    0.6308      3138
"""
import argparse
import os

import tensorflow.compat.v2 as tf
import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics

import allnews_am.processing
import gensim.models
from seqeval.callbacks import F1Metrics


tf.enable_v2_behavior()

file_dir = os.path.dirname(os.path.realpath(__file__))
conll_data_root = os.path.join(file_dir, '../allnews_am/NER_datasets')
COLUMN_TYPES = (
    allnews_am.processing.ConllReader.WORDS,
    allnews_am.processing.ConllReader.POS,
    allnews_am.processing.ConllReader.CHUNK,
    allnews_am.processing.ConllReader.NE,
)


def data_stats(conll_reader, data_suffix=''):
    """Prints basic data statistics.

    Args:
        conll_reader: The ConllReader object.
        data_suffix: Added to the log messages in parentheses.
    """
    train_sentences = conll_reader.iob_sents(column='ne')
    print(f"Number of sentences ({data_suffix}): {len(train_sentences)}")
    train_words = conll_reader.iob_words(column='ne')
    print(f"Number of words ({data_suffix}): {len(train_words)}")
    print(f"Example sentence ({data_suffix}): ", train_sentences[0])


def main(args):
    train_data_reader = allnews_am.processing.ConllReader(
        root=conll_data_root,
        fileids='train.conll03',
        columntypes=COLUMN_TYPES,
    )
    data_stats(train_data_reader, 'train')
    dev_data_reader = allnews_am.processing.ConllReader(
        root=conll_data_root,
        fileids='dev.conll03',
        columntypes=COLUMN_TYPES,
    )
    test_data_reader = allnews_am.processing.ConllReader(
        root=conll_data_root,
        fileids='test.conll03',
        columntypes=COLUMN_TYPES,
    )
    data_stats(dev_data_reader, 'dev')
    train_sentences = train_data_reader.iob_sents(column='ne')
    train_words = train_data_reader.iob_words(column='ne')
    dev_sentences = dev_data_reader.iob_sents(column='ne')
    test_sentences = test_data_reader.iob_sents(column='ne')

    if args.plot_seq_len:
        # Plot the distribution of sentence lengths (in both train and dev).
        sentence_lengths = [len(s) for s in train_sentences + dev_sentences]
        plt.hist(sentence_lengths, bins=50)
        plt.title(f'Tokens per sentence (max = {max(sentence_lengths)})')
        plt.xlabel('Number of tokens (words)')
        plt.ylabel('# samples')
        plt.show()

    # The first 2 entries are reserved for PAD and UNK
    idx2word = ['PAD', 'UNK'] + list(set((t[0] for t in train_words)))
    word2idx = {w: i for i, w in enumerate(idx2word)}

    # The first entry is reserved for PAD
    idx2tag = ['PAD'] + list(set((t[2] for t in train_words)))
    tag2idx = {t: i for i, t in enumerate(idx2tag)}
    print('All NE tags: ', idx2tag)

    idx2char = ['PAD', 'UNK'] + list(
        set([c for w in train_words for c in w[0]]))
    char2idx = {c: i for i, c in enumerate(idx2char)}
    print('Number of unique characters: ', len(char2idx) - 2)

    def encode_tagged_sentences(sentences):
        """Return features and labels from a tagged sequence of sentences.

        Args:
            sentences: The sentences as a sequences of word/pos/IOB tuples.

        Returns:
            A tuple of (encoded_words, encoded_chars, encoded_tags), where of
            of the elements is a sequence. encoded_words is a sequence of ints,
            where the int corresponds ot the index of the word in the vocab
            (word2idx). encoded_chars is a sequence of sequences, where each
            word is split into chars and the chars are encoded as ints, with the
            number corresponding to the index of the char in the char-vocab
            (char2idx). Additionally, each word is zero-padded to the length
            of the longest word in that sentence. encoded_tags is a sequence
            of one-hot-encoded tags, where the encoding is done according to
            tag2idx.
        """
        # Convert each sentence from list of Token to list of word_index
        encoded_words = [
            [word2idx[w[0]] if w[0] in word2idx else 1 for w in s]
            for s in sentences  # 1 is for 'UNK'
        ]
        # Convert each NE tag to a categorical vector.
        encoded_tags = [
            [tf.keras.utils.to_categorical(tag2idx[w[2]], len(tag2idx))
             for w in s] for s in sentences
        ]

        encoded_chars = []
        for s in sentences:
            encoded_chars.append([])
            for w in s:
                encoded_chars[-1].append(
                    [char2idx[c] if c in char2idx else 1 for c in w[0]])
            encoded_chars[-1] = tf.keras.preprocessing.sequence.pad_sequences(
                encoded_chars[-1], padding='post')

        return encoded_words, encoded_chars, encoded_tags

    def prepare_sentence_batches(sentences, batch_size=int(args.batch_size)):
        """Encodes sentences with strings and NE tags to numbers.

        Returns data as tensorflow tensor batches.

        Args:
            sentences: The sentences as sequences of word/pos/IOB tuples.
            batch_size: The size of the batches.

        Returns:
            Shuffled, padded batches of batch_size repeated by the number of
            epochs. The data sample represents a tuple of encoded word/tag paris
            as first element and output_shapes as the 2nd element. The first
            element is a sequence of tuples of form (encoded_word, encoded_tag).
            The 2nd element is a 2-element tuple of output shapes for each of
            these.
        """
        words, chars, tags = encode_tagged_sentences(sentences)

        def data_generator():
            sentence_tag_pairs = zip(zip(words, chars), tags)
            for d in list(sentence_tag_pairs):
                yield d

        output_shapes = (
            (tf.TensorShape([None]), tf.TensorShape([None, None])),
            tf.TensorShape([None, len(tag2idx)])
        )

        data = tf.data.Dataset.from_generator(
            lambda: data_generator(),
            ((tf.int64, tf.int64), tf.int64), output_shapes
        ).shuffle(1000).padded_batch(batch_size, output_shapes)
        return data

    train_batches = prepare_sentence_batches(train_sentences)
    dev_batches = prepare_sentence_batches(dev_sentences)

    # Model definition
    # Word Embeddings
    word_in = tf.keras.layers.Input(shape=(None,))
    if args.pretrained_emb is not None:
        emb_model = gensim.models.fasttext.FastText.load(args.pretrained_emb)
        embedding_matrix = np.zeros((len(idx2word), emb_model.wv.vector_size))
        for i, word in enumerate(idx2word):
            embedding_matrix[i] = emb_model.wv[word]
        emb_word = tf.keras.layers.Embedding(
            input_dim=len(word2idx),
            output_dim=emb_model.wv.vector_size,
            trainable=False,
            weights=[embedding_matrix],
            mask_zero=True)(word_in)
    else:
        emb_word = tf.keras.layers.Embedding(
            input_dim=len(word2idx),
            output_dim=int(args.embedding),
            mask_zero=True)(word_in)

    # Character Embeddings
    char_in = tf.keras.layers.Input(shape=(None, None,))
    emb_char = tf.keras.layers.TimeDistributed(
        tf.keras.layers.Embedding(
            input_dim=len(char2idx),
            output_dim=int(args.char_embedding),
            mask_zero=True))(char_in)
    char_enc = tf.keras.layers.TimeDistributed(
        tf.keras.layers.LSTM(
            units=int(args.char_lstm_size),
            return_sequences=False,
            recurrent_dropout=0.5))(emb_char)

    # Concat Word and Char Embeddings.
    x = tf.keras.layers.concatenate([emb_word, char_enc])
    x = tf.keras.layers.SpatialDropout1D(0.3)(x)
    main_lstm = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(
            units=int(args.lstm_size),
            return_sequences=True,
            recurrent_dropout=0.5))(x)
    out = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(
        len(idx2tag), activation='softmax'))(main_lstm)

    model = tf.keras.models.Model([word_in, char_in], out)
    model.compile(optimizer="adam",
                  loss='categorical_crossentropy')
    model.summary()

    history = model.fit(
        train_batches,
        epochs=int(args.epochs),
        validation_data=dev_batches,
        callbacks=[
            F1Metrics({i: t for i, t in enumerate(idx2tag)},
                      validation_data=dev_batches)
        ],
    )

    def print_classification_score(sentences):
        y_true, y_pred = [], []
        words, chars, tags = encode_tagged_sentences(sentences)
        for w, c, t in zip(words, chars, tags):
            y_true += np.argmax(t, 1).tolist()
            x_true = (np.expand_dims(w, 0), np.expand_dims(c, 0))
            y_pred += np.argmax(
                model.predict_on_batch(x_true), 2).tolist()[0]
        # Index 0 is 'PAD' and should not be in the report.
        print(
            sklearn.metrics.classification_report(
                y_true, y_pred,
                labels=list(range(1, 8)),
                target_names=idx2tag[1:],
                digits=4))

    # These are the "easy scores", where it is not necessary to get the full
    # Named entity to get a non-zero score.
    print("Calculating training set metrics.")
    print_classification_score(train_sentences)
    print("Calculating dev set metrics.")
    print_classification_score(dev_sentences)
    if bool(args.test):
        print("Calculating test set metrics.")
        print_classification_score(test_sentences)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--pretrained_emb',
        default=None,
        # --pretrained_emb=models/fastText_sg_100.model
        help='The path to the pre-trained embeddings. If supplied, uses these '
             'fixed embeddings instead of training them on the fly. This also '
             'overrides the embedding dimension.')
    parser.add_argument(
        '--epochs', default=30,
        help='The number of training epochs (passes through entire dataset).')
    parser.add_argument(
        '--embedding', default=20,
        help='The dimension of the embeddings.')
    parser.add_argument(
        '--char_embedding', default=10,
        help='The dimension of the character embeddings.')
    parser.add_argument(
        '--lstm_size', default=50,
        help='Dimension of the hidden/cell states of the LSTM network.')
    parser.add_argument(
        '--char_lstm_size', default=20,
        help='Dimension of the hidden/cell states of the character LSTM.')
    parser.add_argument(
        '--batch_size', default=32,
        help='Number of examples used in each learning step.')
    parser.add_argument(
        '-test', action='store_true',
        help='If set, calculates and prints scores on the test set.')
    parser.add_argument(
        '-plot_seq_len', action='store_true',
        help='If set, plots distribution of sequence lengths before the '
             'training starts. Would require you to manually close the '
             'matplotlib window to continue the execution of the script.')
    main(parser.parse_args())
