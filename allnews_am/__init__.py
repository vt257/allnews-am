import argparse
import os


def parse_w2v_ft_args(file_dir):
    parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
            '--corpus',
            default=os.path.join(file_dir, '../allnews_am/data/corpus_1679k'),
            help='The path to the corpus to train on.')
    parser.add_argument(
            '--model_name', default='embeddings.model',
            help='The name of the model file (saved in models folder).')
    parser.add_argument('--size', default=100, help='Size of the embedding.')
    parser.add_argument('--window', default=5, help='Context window size.')
    parser.add_argument(
            '--min_count', default=5,
            help='Minimum number of occurrences for word.')
    parser.add_argument(
            '-sg', action='store_true',
            help='If set, will train a skip-gram, otherwise a CBOW.')
    parser.add_argument('--workers', default=4,
                        help='Number of workers.')
    parser.add_argument('--alpha', default=0.1,
                        help='Learning rate.')
    parser.add_argument('--negative', default=10,
                        help='Number of negative samples.')
    parser.add_argument('--epochs', default=30,
                        help='Number of epochs.')
    parser.add_argument('--subsample', default=0.003,
                        help='Sub-sampling rate.')
    parser.add_argument(
            '-add_ner_sents', action='store_true',
            help='If set, will add the NER sentences to the training corpus.')
    return parser.parse_args()
