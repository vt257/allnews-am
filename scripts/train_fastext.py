"""Trains a FastText model."""
import itertools
import logging
import os

import allnews_am
import allnews_am.processing

import gensim.models

logging.basicConfig(
        format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
file_dir = os.path.dirname(os.path.realpath(__file__))


def main(args):
    sentences = gensim.models.word2vec.LineSentence(args.corpus)
    if args.add_ner_sents:
        # Add tagged named entity recognition sentences to training corpus.
        ner_iob_sents = allnews_am.processing.ConllReader(
            root=os.path.join(file_dir, '../allnews_am/NER_datasets'),
            fileids=['train.conll03', 'dev.conll03', 'test.conll03'],
            columntypes=(
                allnews_am.processing.ConllReader.WORDS,
                allnews_am.processing.ConllReader.POS,
                allnews_am.processing.ConllReader.CHUNK,
                allnews_am.processing.ConllReader.NE,
            ),
        ).iob_sents()
        ner_sents = [[w[0] for w in s] for s in ner_iob_sents]
        sentences = [s for s in sentences] + ner_sents
        
    my_model = gensim.models.fasttext.FastText(
        sentences=sentences,
        size=int(args.size),
        window=int(args.window),
        min_count=int(args.min_count),
        workers=int(args.workers),
        alpha=float(args.alpha),
        sample=float(args.subsample),
        negative=int(args.negative),
        sorted_vocab=True,
        min_n=2,
        iter=int(args.epochs))
    
    my_model.save(
        os.path.join(file_dir, '../allnews_am/models/', args.model_name))
    analogy_file = os.path.join(
        file_dir, '../allnews_am/data/yerevann_analogies.txt')
    my_model.wv.evaluate_word_analogies(analogy_file)
    
    
if __name__ == '__main__':
    fast_args = allnews_am.parse_w2v_ft_args(file_dir)
    main(fast_args)
