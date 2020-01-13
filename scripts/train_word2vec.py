"""Trains a word2vec model."""
import os
import logging

import allnews_am
import gensim.models

logging.basicConfig(
        format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
file_dir = os.path.dirname(os.path.realpath(__file__))


def main(args):
    sentences = gensim.models.word2vec.Text8Corpus(args.corpus)

    my_model = gensim.models.word2vec.Word2Vec(
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
    my_model.save(
        os.path.join(file_dir, '../allnews_am/models/', args.model_name))
    analogy_file = os.path.join(
        file_dir, '../allnews_am/data/yerevann_analogies.txt')
    my_model.wv.evaluate_word_analogies(analogy_file)


if __name__ == '__main__':
    word2vec_args = allnews_am.parse_w2v_ft_args(file_dir)
    main(word2vec_args)
