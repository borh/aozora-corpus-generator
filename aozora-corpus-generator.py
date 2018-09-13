import sys
import argparse
import textwrap
import pathlib
import concurrent.futures
import logging

from libs.aozora import (
    make_jis_unicode_map,
    make_ndc_map,
    read_aozora_bunko_list,
    read_author_title_list,
    convert_corpus_file,
    write_metadata_file,
    log,
    stdout_handler,
)


def parse_args():
    '''Parses and returns program arguments as a dictionary.'''
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent('''aozora-corpus-generator extracts given author and book pairs from Aozora Bunko and formats them into (optionally tokenized) plain text files.'''),
        epilog=textwrap.dedent('''Example usage:
python aozora-corpus-generator.py --features 'orth' --author-title-csv 'author-title.csv' --out 'Corpora/Japanese' --parallel''')
    )
    parser.add_argument('--features',
                        nargs='+',
                        help='specify which features should be extracted from morphemes (default=\'orth\')',
                        default=['orth'],
                        required=False)
    parser.add_argument('--features-opening-delim',
                        help='specify opening char to use when outputting multiple features',
                        required=False)
    parser.add_argument('--features-closing-delim',
                        help='specify closing char to use when outputting multiple features',
                        required=False)
    parser.add_argument('--features-separator',
                        help='specify separating char to use when outputting multiple features',
                        required=False)
    parser.add_argument('--author-title-csv',
                        nargs='+',
                        help='one or more UTF-8 formatted CSV input file(s) (default=\'author-title.csv\')',
                        default=['author-title.csv'],
                        required=False)
    parser.add_argument('--aozora-bunko-repository',
                        help='path to the aozorabunko git repository (default=\'aozorabunko/index_pages/list_person_all_extended_utf8.zip\')',
                        default='aozorabunko/index_pages/list_person_all_extended_utf8.zip',
                        required=False)
    parser.add_argument('--out',
                        help='output (plain, tokenized) files into given output directory (default=Corpora)',
                        default='Corpora',
                        required=True)
    parser.add_argument('--all',
                        help='specify if all Aozora Bunko texts should be extracted, ignoring the author-title.csv (default=False)',
                        action='store_true',
                        default=False,
                        required=False)
    parser.add_argument('--min-tokens',
                        help='specify minimum token count to filter files by (default=30000)',
                        default=30000,
                        type=int,
                        required=False)
    parser.add_argument('--no-punc',
                        help='specify if punctuation should be discarded from tokenized version (default=False)',
                        action='store_true',
                        default=False,
                        required=False)
    parser.add_argument('--remove-speech',
                        help='specify if direct speech should be discarded from tokenized version (default=False)',
                        action='store_true',
                        default=False,
                        required=False)
    parser.add_argument('--incremental', # TODO
                        help='do not overwrite existing corpus files (default=False)',
                        action='store_true',
                        default=False,
                        required=False)
    parser.add_argument('--parallel',
                        help='specify if processing should be done in parallel (default=True)',
                        action='store_true',
                        default=False,
                        required=False)
    parser.add_argument('--verbose',
                        help='turns on verbose logging (default=False)',
                        action='store_true',
                        default=True,
                        required=False)

    return vars(parser.parse_args())

if __name__ == '__main__':
    args = parse_args()

    if args['verbose']:
        log.setLevel(logging.DEBUG)
        stdout_handler.setLevel(logging.DEBUG)

    log.addHandler(stdout_handler)

    if not args['author_title_csv'] and not args['all']:
        print('Please specify the "--author-title-csv"  or "--all" option.\nAborting.')
        sys.exit(1)

    gaiji_tr = make_jis_unicode_map('jisx0213-2004-std.txt')
    ndc_tr = make_ndc_map()

    pathlib.Path(args['out'] + '/Tokenized').mkdir(parents=True, exist_ok=True)
    pathlib.Path(args['out'] + '/Plain').mkdir(parents=True, exist_ok=True)

    aozora_db = read_aozora_bunko_list(args['aozora_bunko_repository'], ndc_tr)

    files, metadata = [], []
    if args['all']:
        for author_ja, titles in aozora_db.items():
            for title, title_dict in titles.items():
                files.append(('Aozora Bunko', title_dict['file_name'], title_dict['file_path']))
                metadata.append({
                    'corpus': 'Aozora Bunko',
                    'corpus_id': title_dict['file_path'],
                    'author': title_dict['author'],
                    'title': title,
                    'brow': '',
                    'genre': '',
                    'narrative_perspective': '',
                    'comments': ''
                })
    else:
        for csv_path in args['author_title_csv']:
            fs, db = read_author_title_list(aozora_db, csv_path)
            files.extend(fs)
            metadata.extend(db)

    rejected_files = set()

    token_counts = {}

    if args['parallel']:
        with concurrent.futures.ProcessPoolExecutor() as executor:
            futures = [executor.submit(convert_corpus_file,
                                       corpus, file_name, file_path,
                                       args['out'], gaiji_tr,
                                       args['features'],
                                       args['no_punc'],
                                       args['remove_speech'],
                                       args['min_tokens'],
                                       args['features_separator'],
                                       args['features_opening_delim'],
                                       args['features_closing_delim'])
                       for (corpus, file_name, file_path) in files]
            for future in concurrent.futures.as_completed(futures):
                try:
                    file_name, file_path, prefix, token_count, rejected = future.result()
                    token_counts[file_path] = token_count
                    if rejected:
                        rejected_files.add(file_path)
                        log.warn(f'{file_name} rejected because it contains < {args["min_tokens"]} tokens.')
                    else:
                        log.info('{} => {}/{{ Tokenized, Plain }}/{}.txt'.format(file_path, prefix, file_name))
                except Exception:
                    log.error('Process {} failed: {}'.format(future, future.result()))
    else:
        for corpus, file_name, file_path in files:
            _, _, _, token_count, rejected = convert_corpus_file(
                corpus,
                file_name,
                file_path,
                args['out'],
                gaiji_tr,
                args['features'],
                args['no_punc'],
                args['remove_speech'],
                args['min_tokens'],
                args['features_separator'],
                args['features_opening_delim'],
                args['features_closing_delim']
            )
            token_counts[file_path] = token_count
            if rejected:
                rejected_files.add(file_path)
                log.warn(f'{file_name} rejected because it contains < {args["min_tokens"]} tokens.')
            else:
                log.info('{} => {}/{{ Tokenized, Plain }}/{}.txt'.format(file_path, args['out'], file_name))

    rejected_indeces = {idx for idx, (_, _, file_path) in enumerate(files)
                        if file_path in rejected_files}

    files = [file for idx, file in enumerate(files)
             if idx not in rejected_indeces]

    for m in metadata:
        m['token_count'] = token_counts[m['corpus_id']]

    metadata = [m for idx, m in enumerate(metadata)
                if idx not in rejected_indeces]

    write_metadata_file(files, metadata, aozora_db, args['out'])
