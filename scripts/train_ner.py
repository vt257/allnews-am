"""Train an NER model."""
import os

import tensorflow.compat.v2 as tf
import matplotlib.pyplot as plt

import allnews_am.processing

tf.enable_v2_behavior()

file_dir = os.path.dirname(os.path.realpath(__file__))
conll_data_root = os.path.join(file_dir, '../allnews_am/NER_datasets')
COLUMN_TYPES = (
    allnews_am.processing.ConllReader.WORDS,
    allnews_am.processing.ConllReader.POS,
    allnews_am.processing.ConllReader.CHUNK,
    allnews_am.processing.ConllReader.NE,
)

BATCH_SIZE = 32  # Number of examples used in each learning step.
EPOCHS = 20  # Number of passes through entire dataset.
EMBEDDING = 50  # Dimension of word embedding vector.
LSTM_SIZE = 32  # Dimension of the hidden/cell states of the LSTM network.


def f1_score(y_true, y_pred):
    """Get the f1 score for keras."""
    true_positives = tf.keras.backend.sum(tf.keras.backend.round(
        tf.keras.backend.clip(y_true * y_pred, 0, 1)))
    possible_positives = tf.keras.backend.sum(
        tf.keras.backend.round(tf.keras.backend.clip(y_true, 0, 1)))
    predicted_positives = tf.keras.backend.sum(
        tf.keras.backend.round(tf.keras.backend.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives +
                                  tf.keras.backend.epsilon())
    recall = true_positives / (possible_positives +
                               tf.keras.backend.epsilon())
    f1_val = 2 * (precision * recall)/(precision + recall +
                                       tf.keras.backend.epsilon())
    return f1_val


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


def main():
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

    def prepare_sentence_batches(sentences, batch_size=BATCH_SIZE):
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
    x = tf.keras.layers.Embedding(
        input_dim=len(word2idx),
        output_dim=EMBEDDING,
        mask_zero=True)(input_layer)
    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(
            units=LSTM_SIZE, return_sequences=True, recurrent_dropout=0.1))(x)
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(
        len(idx2tag), activation='softmax'))(x)

    model = tf.keras.models.Model(input_layer, x)
    model.compile(
        optimizer="adam", loss='binary_crossentropy', metrics=[f1_score])
    model.summary()

    history = model.fit(
        train_batches,
        epochs=EPOCHS,
        validation_data=dev_batches,
    )


if __name__ == '__main__':
    main()
