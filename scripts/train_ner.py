"""Train an NER model.

Below are some scores for the "difficult test", where the result is only
satisfactory if the full named entity is recognized.

30 epochs with embedding/LSTM size 100/32 without any pre-trained embeddings:
 - f1: 56.34
           precision    recall  f1-score   support

      LOC     0.6246    0.5868    0.6051      1849
      PER     0.6165    0.4925    0.5476       865
      ORG     0.4183    0.4104    0.4143       424

micro avg     0.5925    0.5370    0.5634      3138
macro avg     0.5945    0.5370    0.5635      3138

30 epochs with embedding/LSTM size 100/32 with pre-trained embeddings:
fastText on 100k news articles.
 - f1: 57.85
           precision    recall  f1-score   support

      LOC     0.6816    0.6160    0.6472      1849
      PER     0.6523    0.5075    0.5709       865
      ORG     0.3053    0.3420    0.3226       424

micro avg     0.6112    0.5491    0.5785      3138
macro avg     0.6227    0.5491    0.5823      3138
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
    data_stats(dev_data_reader, 'dev')
    train_sentences = train_data_reader.iob_sents(column='ne')
    train_words = train_data_reader.iob_words(column='ne')
    dev_sentences = dev_data_reader.iob_sents(column='ne')

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
    test_word = 'նա'
    print(f'The word {test_word} is identified by the index: '
          f'{word2idx[test_word]}')

    # The first entry is reserved for PAD
    idx2tag = ['PAD'] + list(set((t[2] for t in train_words)))
    tag2idx = {t: i for i, t in enumerate(idx2tag)}
    print('All NE tags: ', idx2tag)
    test_tag = 'B-ORG'
    print(f'The labels {test_tag} is identified by the index: '
          f'{tag2idx[test_tag]}')

    def prepare_sentence_batches(sentences, batch_size=int(args.batch_size)):
        """Encodes sentences with strings and NE tags to numbers.

        Returns data as tensorflow tensor batches.

        Args:
            sentences: The sentences as word/pos/IOB tuples.
            batch_size: The size of the batches.

        Returns:
            Shuffled, padded batches of batch_size repeated by the number of
            epochs. The data sample represents a tuple of encoded word/tag paris
            as first element and output_shapes as the 2nd element. The first
            element is a sequence of tuples of form (encoded_word, encoded_tag).
            The 2nd element is a 2-element tuple of output shapes for each of
            these.
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

        def data_generator():
            # Combine for the dataset.
            sentence_tag_pairs = zip(encoded_words, encoded_tags)
            for d in list(sentence_tag_pairs):
                yield d

        output_shapes = (
            tf.TensorShape([None]),
            tf.TensorShape([None, len(tag2idx)])
        )

        data = tf.data.Dataset.from_generator(
            lambda: data_generator(),
            (tf.int64, tf.int64), output_shapes
        ).shuffle(1000).padded_batch(batch_size, output_shapes)
        return data

    train_batches = prepare_sentence_batches(train_sentences)
    dev_batches = prepare_sentence_batches(dev_sentences)

    # Model definition
    input_layer = tf.keras.layers.Input(shape=(None,))
    if args.pretrained_emb is not None:
        emb_model = gensim.models.fasttext.FastText.load(args.pretrained_emb)
        embedding_matrix = np.zeros((len(idx2word), emb_model.wv.vector_size))
        for i, word in enumerate(idx2word):
            embedding_matrix[i] = emb_model.wv[word]
        x = tf.keras.layers.Embedding(
            input_dim=len(word2idx),
            output_dim=emb_model.wv.vector_size,
            trainable=False,
            weights=[embedding_matrix],
            mask_zero=True)(input_layer)
    else:
        x = tf.keras.layers.Embedding(
            input_dim=len(word2idx),
            output_dim=int(args.embedding),
            mask_zero=True)(input_layer)
    x = tf.keras.layers.Dropout(float(args.emb_dropout))(x)
    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(
            units=int(args.lstm_size),
            return_sequences=True,
            recurrent_dropout=float(args.recurrent_dropout)))(x)
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(
        len(idx2tag), activation='softmax'))(x)

    model = tf.keras.models.Model(input_layer, x)
    model.compile(optimizer="adam", loss='binary_crossentropy')
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
        for i, sentence in enumerate(sentences):
            y_true += [tag2idx[w[2]] for w in sentence]
            x_true = [
                [word2idx[w[0]] if w[0] in word2idx else 1 for w in sentence]
            ]
            y_pred += np.argmax(
                model.predict_on_batch(np.asarray(x_true)), 2).tolist()[0]
        print(sklearn.metrics.classification_report(
            y_true, y_pred, labels=list(range(1, 8)), target_names=idx2tag[1:]))

    # These are the "easy scores", where it is not necessary to get the full
    # Named entity to get a non-zero score.
    print("Calculating training set metrics.")
    print_classification_score(train_sentences)
    print("Calculating dev set metrics.")
    print_classification_score(dev_sentences)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--pretrained_emb',
        default=None,
        # allnews_am/models/fastText_sg_100.model
        help='The path to the pre-trained embeddings. If supplied, uses these '
             'fixed embeddings instead of training them on the fly. This also '
             'overrides the embedding dimension.')
    parser.add_argument(
        '--embedding', default=100,
        help='The dimension of the embeddings.')
    parser.add_argument(
        '--epochs', default=30,
        help='The number of training epochs (passes through entire dataset).')
    parser.add_argument(
        '--lstm_size', default=32,
        help='Dimension of the hidden/cell states of the LSTM network.')
    parser.add_argument(
        '--emb_dropout', default=0.2,
        help='The dropout applied right after the embedding layer.')
    parser.add_argument(
        '--recurrent_dropout', default=0.2,
        help='The recurrent dropout of the LSTM layer.')
    parser.add_argument(
        '--batch_size', default=32,
        help='Number of examples used in each learning step.')
    parser.add_argument(
        '-plot_seq_len', action='store_true',
        help='If set, plots distribution of sequence lengths before the '
             'training starts. Would require you to manually close the '
             'matplotlib window to continue the execution of the script.')
    main(parser.parse_args())
