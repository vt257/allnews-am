"""Trains a word2vec model."""
import argparse
import os
import logging

import gensim.models
from gensim.models.fasttext import FastText

logging.basicConfig(
        format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
file_dir = os.path.dirname(os.path.realpath(__file__))


def main(args):
    sentences = gensim.models.word2vec.Text8Corpus(args.corpus)

    my_model = FastText(sentences = sentences,
         size=int(args.size),
        window=int(args.window),
        min_count=int(args.min_count),
        workers=int(args.workers),
        alpha=float(args.alpha),
        sample=float(args.subsample),
        negative=int(args.negative),
        sorted_vocab=True,
        iter=int(args.epochs))
    
    my_model.save(
        os.path.join(file_dir, '../allnews_am/models/', args.model_name))
    analogy_file = os.path.join(
        file_dir, '../allnews_am/data/yerevann_analogies.txt')
    my_model.wv.evaluate_word_analogies(analogy_file)
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
            '--corpus',
            default=os.path.join(file_dir, '../allnews_am/data/corpus'),
            help='The path to the corpus to train on.')
    parser.add_argument(
            '--model_name', default='word2vec.model',
            help='The name of the model file (saved in models folder).')
    parser.add_argument('--size', default=100, help='Size of the embedding.')
    parser.add_argument('--window', default=15, help='Context window size.')
    parser.add_argument(
            '--min_count', default=5,
            help='Minimum number of occurrences for word.')
    parser.add_argument('--workers', default=12,
                        help='Number of threads.')
    parser.add_argument('--alpha', default=0.025,
                        help='Learning rate.')
    parser.add_argument('--negative', default=15,
                        help='Number of negative samples.')
    parser.add_argument('--epochs', default=30,
                        help='Number of epochs.')
    parser.add_argument('--subsample', default=0.0001,
                        help='Sub-sampling rate.')
    fast_args = parser.parse_args()
    main(fast_args)

