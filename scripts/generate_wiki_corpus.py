"""Generates a corpus file from the processed Wikipedia json."""
import argparse
import json
import multiprocessing
import os
import tqdm

import allnews_am.processing

file_dir = os.path.dirname(os.path.realpath(__file__))
STANDARDIZE = True


def process_json(json_dict, standardize=False):
    """Processes a single json dict into a sequence of sentences."""
    title_str = json_dict['title'].strip()
    text_str = json_dict['text'].strip()
    processed_sentences = []
    for string in title_str, text_str:
        if not string:
            continue
        tokenized = allnews_am.processing.tokenize(string)
        if standardize:
            tokenized = allnews_am.processing.standardize(tokenized)
        for sentence in tokenized:
            # One sentence per line.
            concatenated = ' '.join(sentence)
            processed_sentences.append(concatenated)
    return processed_sentences


def process_single_file(json_full_path):
    print(f'Processing {json_full_path}')
    with open(json_full_path, 'r', encoding='utf-8') as f:
        all_lines = f.readlines()
    with open(json_full_path + "corpus", 'w', encoding='utf-8') as corpus_f:
        for line in tqdm.tqdm(all_lines):
            processed_article = process_json(
                json.loads(line), standardize=STANDARDIZE)
            for sentence in processed_article:
                corpus_f.write(f'{sentence}\n')


def main(args):
    wiki_json_files = [
        os.path.join(args.wiki_folder, file_name)for file_name
        in os.listdir(args.wiki_folder) if 'corpus' not in file_name
    ]
    workers = 4
    with multiprocessing.Pool(workers) as p:
        list(p.imap(process_single_file, wiki_json_files))

    # Join corpus files
    wiki_corpus_files = [
        os.path.join(args.wiki_folder, file_name)for file_name
        in os.listdir(args.wiki_folder) if 'corpus' in file_name
    ]
    write_file_name = os.path.join(args.wiki_folder, 'hywiki_all')
    all_content = ''
    for wiki_corpus_file in wiki_corpus_files:
        with open(wiki_corpus_file, 'r', encoding='utf-8') as f:
            all_content += f.read()
    with open(write_file_name, 'w', encoding='utf-8') as f:
        f.write(all_content)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--wiki_folder",
        default=os.path.join(file_dir, '../data/wiki_json'),
        help="The path to the folder with processed wiki json files.")
    main(parser.parse_args())
