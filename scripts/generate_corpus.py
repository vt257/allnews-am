"""Generates the pre-processed text corpus from a MySQL database."""
import argparse
import os
import tqdm

import allnews_am.db
import allnews_am.processing

file_dir = os.path.dirname(os.path.realpath(__file__))


def main(args):
    mysql_db = allnews_am.db.MySQL()
    all_news = mysql_db.fetch_news(
        offset=int(args.offset), limit=int(args.limit))

    with open(args.corpus_filename, 'w', encoding='utf-8') as corpus_file:
        for title, text in tqdm.tqdm(all_news):
            for string in [title, text]:
                if string is None:
                    continue
                tokenized = allnews_am.processing.tokenize(string)
                standardized = allnews_am.processing.standardize(tokenized)
                for sentence in standardized:
                    # One sentence per line.
                    concatenated = ' '.join(sentence)
                    corpus_file.write(f'{concatenated}\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
            "--corpus_filename",
            default=os.path.join(file_dir, "../data/corpus"),
            help="The path to save the corpus to.")
    parser.add_argument(
        "--offset",
        default=0,
        help="The offset for the sql query.")
    parser.add_argument(
        "--limit",
        default=allnews_am.db.MYSQL_MAX_LIMIT,
        help="The limit for the sql query.")
    corpus_args = parser.parse_args()
    main(corpus_args)
