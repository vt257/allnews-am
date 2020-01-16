import os

import allnews_am.processing

file_dir = os.path.dirname(os.path.realpath(__file__))
conll_data_root = os.path.join(file_dir, '../allnews_am/NER_datasets')
COLUMN_TYPES = (
    allnews_am.processing.ConllReader.WORDS,
    allnews_am.processing.CognllReader.POS,
    allnews_am.processing.ConllReader.CHUNK,
    allnews_am.processing.ConllReader.NE,
)


def main():
    train_data_reader = allnews_am.processing.ConllReader(
        root=conll_data_root,
        fileids='train.conll03',
        columntypes=COLUMN_TYPES,
    )
    print(
        f"Number of sentences: {len(train_data_reader.iob_sents(column='ne'))}")
    print(
        f"Number of words: {len(train_data_reader.iob_words(column='ne'))}")
    print("Example sentence:")
    print(train_data_reader.iob_sents(column='ne')[0])


if __name__ == '__main__':
    main()
