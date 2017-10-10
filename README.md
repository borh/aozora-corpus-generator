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

You must install [MeCab]() and [UniDic]().

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
pip install natto-py jaconv lxml
```

# Usage

    usage: aozora-corpus-generator.py [-h] [--features FEATURES [FEATURES ...]]
                                      --author-title-csv AUTHOR_TITLE_CSV
                                      [AUTHOR_TITLE_CSV ...]
                                      [--aozora-bunko-repository AOZORA_BUNKO_REPOSITORY]
                                      --out OUT [--parallel]
    aozora-corpus-generator extracts given author and book pairs from Aozora Bunko and formats them into (optionally tokenized) plain text files.
    optional arguments:
      -h, --help            show this help message and exit
      --features FEATURES [FEATURES ...]
                            specify which features should be extracted from
                            morphemes (default is 'orth')
      --author-title-csv AUTHOR_TITLE_CSV [AUTHOR_TITLE_CSV ...]
                            one or more UTF-8 formatted CSV input file(s) (default
                            is 'author-title.csv')
      --aozora-bunko-repository AOZORA_BUNKO_REPOSITORY
                            path to the aozorabunko git repository
      --out OUT             output (plain, tokenized) files into given output
                            directory
      --parallel            specify if processing should be done in parallel
                            (default=True)
      --verbose             turns on verbose logging (default=False)

    Example usage:
    python aozora-corpus-generator.py --features 'orth' --author-title-csv 'author-title.csv' --out 'Corpora/Japanese' --parallel
