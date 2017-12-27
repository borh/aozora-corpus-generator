# Aozora Bunko Corpus Generator

Generates plain or tokenized text files from the [Aozora Bunko](http://www.aozora.gr.jp/) [[English](https://en.wikipedia.org/wiki/Aozora_Bunko)] for use in corpus-based studies.

# Goals

Primarily for use in an upcoming research project.

# Requirements

## Aozora Bunko Repository

**WARNING**:
Currently, the tool requires a [checked-out repository of the Aozora Bunko](https://github.com/aozorabunko/aozorabunko).
A git clone will take up to several hours and take up **12**GB of space.
Future versions will ease this requirement.

## Native

You must install [MeCab](https://github.com/taku910/mecab) and [UniDic](https://osdn.net/projects/unidic/).

On Debian-based distros, the command below should suffice:

```bash
sudo apt get install -y mecab libmecab-dev unidic-mecab
```

MacOS users can install the native dependencies with:

```bash
brew install mecab mecab-unidic
```

## Python

Python 3 is required. All testing is done on the latest stable version (currently 3.6.2), but a slightly older version should also work.
Native dependencies must be installed before installing the Python dependencies (natto-py needs MeCab).

This project uses [pipenv](https://github.com/kennethreitz/pipenv).
For existing users, the command below should suffice:

```bash
pipenv install
pipenv shell
```

For those using `pip`, you can install all the dependencies using the command below:

```bash
pip install natto-py jaconv lxml html5_parser
```

# Usage

Clone the repository and run:

```bash
git clone https://github.com/borh/aozora-corpus-generator.git
cd aozora-corpus-generator
pipenv install
pipenv shell
python aozora-corpus-generator.py --features 'orth' --author-title-csv 'author-title.csv' --out 'Corpora/Japanese' --parallel
```

## Parameters

```bash
python aozora-corpus-generator.py --help
```

    usage: aozora-corpus-generator.py [-h] [--features FEATURES [FEATURES ...]]
                                      [--features-opening-delim FEATURES_OPENING_DELIM]
                                      [--features-closing-delim FEATURES_CLOSING_DELIM]
                                      [--author-title-csv AUTHOR_TITLE_CSV [AUTHOR_TITLE_CSV ...]]
                                      [--aozora-bunko-repository AOZORA_BUNKO_REPOSITORY]
                                      --out OUT [--all] [--min-tokens MIN_TOKENS]
                                      [--no-punc] [--incremental] [--parallel]
                                      [--verbose]
    aozora-corpus-generator extracts given author and book pairs from Aozora Bunko and formats them into (optionally tokenized) plain text files.
    optional arguments:
      -h, --help            show this help message and exit
      --features FEATURES [FEATURES ...]
                            specify which features should be extracted from
                            morphemes (default='orth')
      --features-opening-delim FEATURES_OPENING_DELIM
                            specify opening char to use when outputting multiple
                            features
      --features-closing-delim FEATURES_CLOSING_DELIM
                            specify closing char to use when outputting multiple
                            features
      --author-title-csv AUTHOR_TITLE_CSV [AUTHOR_TITLE_CSV ...]
                            one or more UTF-8 formatted CSV input file(s)
                            (default='author-title.csv')
      --aozora-bunko-repository AOZORA_BUNKO_REPOSITORY
                            path to the aozorabunko git repository (default='aozor
                            abunko/index_pages/list_person_all_extended_utf8.zip')
      --out OUT             output (plain, tokenized) files into given output
                            directory (default=Corpora)
      --all                 specify if all Aozora Bunko texts should be extracted,
                            ignoring the author-title.csv (default=False)
      --min-tokens MIN_TOKENS
                            specify minimum token count to filter files by
                            (default=30000)
      --no-punc             specify if punctuation should be discarded from
                            tokenized version (default=False)
      --incremental         do not overwrite existing corpus files (default=False)
      --parallel            specify if processing should be done in parallel
                            (default=True)
      --verbose             turns on verbose logging (default=False)
    Example usage:
    python aozora-corpus-generator.py --features 'orth' --author-title-csv 'author-title.csv' --out 'Corpora/Japanese' --parallel
