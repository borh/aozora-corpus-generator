import argparse
import textwrap
import pathlib
import concurrent.futures

import re
import unicodedata
import jaconv
import csv
from collections import defaultdict
from zipfile import ZipFile
from io import TextIOWrapper
from lxml import html
from natto import MeCab


def normalize_japanese_text(s):
    '''
    Normalizes to NFKC Unicode norm and converts all half-width
    alphanumerics and symbols to full-width. The latter step is needed
    for correct use with the UniDic dictionary used with MeCab.
    '''
    return jaconv.h2z(unicodedata.normalize('NFKC', s),
                      kana=False,
                      digit=True,
                      ascii=True).replace('.', '．').replace(',', '，')


def split_sentence_ja(s):
    '''
    Splits Japanese sentence on common sentence delimiters. Some care
    is taken to prevent false positives, but more sophisticated
    (non-regex) logic would be required to implement a more robust
    solution.
    '''
    delimiters = r'[!\?。．！？…]'
    closing_quotations = r'[\)）」』】］〕〉》\]]'
    return [s.strip() for s in re.sub(r'({}+)(?!{})'.format(delimiters, closing_quotations),
                                      r'\1\n',
                                      s).splitlines()]


def text_to_tokens(text):
    '''
    Returns a sequence of sentences, each comprising a sequence of
    tokens. Must be subsumed into non-lazy collection.
    '''
    unidic_features = ['pos1', 'pos2', 'pos3', 'pos4', 'cType',
                       'cForm', 'lForm', 'lemma', 'orth', 'pron', 'orthBase',
                       'pronBase', 'goshu', 'iType', 'iForm', 'fType', 'fForm']

    with MeCab() as mecab:
        for sentence in split_sentence_ja(text):
            tokens = []
            for node in mecab.parse(sentence, as_nodes=True):
                if not node.is_eos():
                    token = dict(zip(unidic_features, node.feature.split(',')))
                    if len(token) == 6:  # UNK
                        token['orth'] = node.surface
                        token['orthBase'] = node.surface
                        token['lemma'] = node.surface
                        tokens.append(token)
                    else:
                        tokens.append(token)
            yield tokens


def wakati(text):
    '''
    Returns a sequence of sentences comprised of whitespace separated tokens.
    '''
    for sentence in text_to_tokens(text):
        yield ' '.join(token['orth'] for token in sentence)


def tokenize(text, features):
    '''
    Returns a sequence of sentences comprised of whitespace separated
    tokens. Supports encoding tokens with other POS or morphological
    annotations.
    '''
    for sentence in text_to_tokens(text):
        yield ' '.join('/'.join(token[feature] for feature in features)
                       for token in sentence)


from pprint import pprint
def read_aozora_bunko_list(path):
    ''''''
    d = defaultdict(dict)
    url_rx = re.compile(r'http://www\.aozora\.gr\.jp/cards/(\d+)/(.+)')
    with ZipFile(path) as z:
        with z.open('list_person_all_extended_utf8.csv', 'r') as f:
            for row in csv.DictReader(TextIOWrapper(f)):
                # Some works have versions in both new- and old-style
                # kana. As we are only interested in the new-style
                # version, we skip the old one while keeping only
                # old-style works.
                if row['文字遣い種別'] == '旧字旧仮名':
                    pass

                author = row['姓'] + row['名']
                title = row['作品名']

                try:
                    match = url_rx.match(row['XHTML/HTMLファイルURL'])
                    id = match.group(1)
                    file_path = match.group(2)
                except AttributeError:
                    print('No XML, ignoring {}, {}'.format(author, title))

                d[author][title] = {
                    'file_path': 'aozorabunko/cards/{}/{}'.format(id, file_path),
                    'file_name': '{}_{}_{}'.format(  # TODO Do we need to shorthen these?
                        row['姓ローマ字'],
                        row['名ローマ字'][0:1],
                        jaconv.kana2alphabet(jaconv.kata2hira(row['作品名読み'][0:5].replace('・', '_')))
                    )
                }
    return d


def read_author_title_list(aozora_db, path):
    ''''''
    corpus_files = []
    with open(path, newline='') as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                match = aozora_db[row['author'].replace(' ', '')][row['title']]
                corpus_files.append((match['file_name'], match['file_path']))
            except KeyError:
                print('{} not in Aozora Bunko DB'.format(row))
    return corpus_files


def read_aozora_bunko_xml(path):
    ''''''
    doc = html.parse(path)
    body = doc.xpath(".//div[@class='main_text']")[0]
    if len(body) == 0:
        print('WARN: look into this file')
        body = doc.xpath(".//div")[0]
    # Remove ruby and notes:
    for e in body.xpath(".//span[@class='notes'] | .//rp | .//rt"):
        e.drop_tree()

    text = re.sub(r'[\n\s]+', '\n', ''.join(body.itertext()).strip(), re.M)

    return [list(wakati(paragraph)) for paragraph in text.splitlines()]


def write_corpus_file(sentences, file_name, prefix):
    ''''''
    with open('{}/{}.txt'.format(prefix, file_name), 'w') as f:
        for sentence in sentences:
            f.write(sentence + '\n')


def convert_corpus_file(file_name, file_path, prefix):
    write_corpus_file(read_aozora_bunko_xml(file_path), file_name, prefix)


def parse_args():
    '''Parses and returns program arguments as dictionary.'''
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent('''aozora-corpus-generator extracts given author and book pairs from Aozora Bunko and formats them into (optionally tokenized) plain text files.'''),
        epilog=textwrap.dedent('''Example usage:
$ python aozora-corpus-generator.py --features 'orth' --author-title-csv 'author-title.csv' --aozora-bunko-repository 'aozorabunko' --out 'Corpora' ''')
    )
    parser.add_argument('--features',
                        nargs='+',
                        help='specify which features should be extracted from morphemes (default is \'orth\')',
                        default=['orth'],
                        required=False)
    parser.add_argument('--author-title-csv',
                        nargs='+',
                        help='one or more UTF-8 formatted CSV input file(s) (default is \'author-title.csv\')',
                        default=['author-title.csv'],
                        required=True)
    parser.add_argument('--aozora-bunko-repository',
                        help='path to the aozorabunko git repository',
                        default='aozorabunko/index_pages/list_person_all_extended_utf8.zip',
                        required=False)
    parser.add_argument('--out',
                        help='output (plain, tokenized) files into given output directory',
                        default='Corpora',
                        required=True)
    parser.add_argument('--parallel',
                        help='specify if processing should be done in parallel (default=True)',
                        action='store_true',
                        default=False,
                        required=False)

    return vars(parser.parse_args())


if __name__ == '__main__':
    args = parse_args()

    pprint(args)

    pathlib.Path(args['out']).mkdir(parents=True, exist_ok=True)

    aozora_db = read_aozora_bunko_list(args['aozora_bunko_repository'])
    files = [f for csv_path in args['author_title_csv']
             for f in read_author_title_list(aozora_db, csv_path)]

    print(files)

    if args['parallel']:
        with concurrent.futures.ProcessPoolExecutor() as executor:
            futures = [executor.submit(convert_corpus_file, file_name, file_path, args['out']) for (file_name, file_path) in files]
    else:
        for file_name, file_path in files:
            convert_corpus_file(file_name, file_path, args['out'])
            print('{} => {}/{{ Tokenized, Plain }}/{}.txt'.format(file_path, args['out'], file_name))
