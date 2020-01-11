"""Generates the pre-processed text corpus from a MySQL database."""
import argparse
import os
import tqdm

import allnews_am.db
import allnews_am.processing


file_dir = os.path.dirname(os.path.realpath(__file__))


def main(args):
    mysql_db = allnews_am.db.MySQL()
    all_news = mysql_db.fetch_news()

    with open(args.corpus_filename, 'w', encoding='utf-8') as corpus_file:
        for title, text in tqdm.tqdm(all_news):
            for string in [title, text]:
                tokenized = allnews_am.processing.tokenize(string)
                concatenated = ' '.join(
                    [' '.join(sentence) for sentence in tokenized])
                corpus_file.write(f'{concatenated}\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
            "--corpus_filename",
            default=os.path.join(file_dir, "../allnews_am/data/corpus"),
            help="The path to save the corpus to.")
    corpus_args = parser.parse_args()
    main(corpus_args)
