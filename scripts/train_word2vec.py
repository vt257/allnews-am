"""Trains a word2vec model."""
import argparse
import os
import logging

import gensim.models

logging.basicConfig(
        format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
file_dir = os.path.dirname(os.path.realpath(__file__))


def main(args):
    sentences = gensim.models.word2vec.Text8Corpus(args.corpus)

    my_model = gensim.models.Word2Vec(
            sentences,
            size=int(args.size),
            window=int(args.window),
            min_count=int(args.min_count),
            sg=bool(args.sg),
            workers=int(args.workers),
            alpha=float(args.alpha),
            sample=float(args.subsample),
            negative=int(args.negative),
            compute_loss=True,
            sorted_vocab=True,
            iter=int(args.epochs))
    my_model.save(args.model_file)
    analogy_file = os.path.join("allnews_am/data", "yerevann_analogies.txt")
    my_model.wv.accuracy(analogy_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
            "--corpus",
            default=os.path.join(file_dir, "../allnews_am/data/corpus"),
            help="The path to the corpus to train on.")
    parser.add_argument(
            "--model_file",
            default=os.path.join("allnews_am/models", "word2vec.model"),
            help="The path to the corpus to train on.")
    parser.add_argument("--size", default=256, help="Size of the embedding.")
    parser.add_argument("--window", default=8, help="Context window size.")
    parser.add_argument(
            "--min_count", default=5,
            help="Minimum number of occurrences for word.")
    parser.add_argument(
            "-sg", action="store_true",
            help="If set, will train a skip-gram, otherwise a CBOW.")
    parser.add_argument("--workers", default=4,
                        help="Number of workers.")
    parser.add_argument("--alpha", default=0.025,
                        help="Learning rate.")
    parser.add_argument("--negative", default=10,
                        help="Number of negative samples.")
    parser.add_argument("--epochs", default=30,
                        help="Number of epochs.")
    parser.add_argument("--subsample", default=0.001,
                        help="Sub-sampling rate.")
    word2vec_args = parser.parse_args()
    main(word2vec_args)
