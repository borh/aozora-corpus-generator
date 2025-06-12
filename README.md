# Aozora Bunko Corpus Generator

Generates plain or tokenized text files from the [Aozora Bunko](http://www.aozora.gr.jp/) [[English](https://en.wikipedia.org/wiki/Aozora_Bunko)] for use in corpus-based studies.

# Requirements

## Aozora Bunko Repository (repo-mode only)

**WARNING**:
Currently, the tool requires a [checked-out repository of the Aozora Bunko](https://github.com/aozorabunko/aozorabunko) for batch extraction.
A git clone will take up to several hours and consume ~25 GB.

**NOTE**: If you only need plain‐text from a single XHTML file (using `--xhtml-file`), you do *not* need any Aozora Bunko repository or CSV database.

## Native

You must install [MeCab](https://github.com/taku910/mecab) and [UniDic](https://osdn.net/projects/unidic/).

On Debian-based distros, the command below should suffice:

```bash
sudo apt install -y mecab libmecab-dev unidic-mecab
```

MacOS users can install the native dependencies with:

```bash
brew install mecab mecab-unidic
```

For nix users a `flake.nix` file is provided:
```bash
nix develop
```

## Python

Python ≥ 3.10 is required (we test on 3.12). Install the package (and its console script) in editable mode (or just use uv):

```bash
pip install -e .
```

# Usage

Clone the repository and run:

```bash
git clone https://github.com/borh/aozora-corpus-generator.git
cd aozora-corpus-generator
pip install -e .

# batch extraction example:
uv run aozora-corpus-generator \
  --features orth \
  --author-title-csv author-title.csv \
  --min-tokens 10 \
  --out Corpora/Japanese \
  --parallel
```

If you just have one Aozora XHTML/HTML file and want its plain text (or sentences or tokens):

```bash
# raw text
uv run aozora-corpus-generator --xhtml-file path/to/Example.xhtml

# one sentence per line (blank line between paragraphs)
uv run aozora-corpus-generator --xhtml-file path/to/Example.xhtml \
  --split-sentences

# tokenized sentences (space-separated tokens)
uv run aozora-corpus-generator --xhtml-file path/to/Example.xhtml \
  --tokenized-only
```

This writes the plain text to stdout; to save to a file:

```bash
uv run aozora-corpus-generator --xhtml-file path/to/cards/12345/Example.html > Example.txt
```

You may also use `uv run` to run the program:

```bash
uv run aozora-corpus-generator --features orth --author-title-csv author-title.csv --out Corpora/Japanese --parallel
```

Warning: this will use all available threads.

# Programmatic Usage

You can also import and call the core functions from your own Python code:

```python
from aozora_corpus_generator.aozora import (
    make_ndc_map,
    read_aozora_bunko_list,
    read_aozora_bunko_xml,
)

## Repo-mode:
# prepare NDC mapping
ndc_map = make_ndc_map()

# load Aozora index
aozora_db = read_aozora_bunko_list(
    "aozorabunko/index_pages/list_person_all_extended_utf8.zip",
    ndc_map,
)

## Single-file mode:
meta = aozora_db["芥川龍之介"]["羅生門"]
path = meta["file_path"] # Or specify any xhtml file with its path

text, paragraphs, token_count = read_aozora_bunko_xml(
    path,
    features=["orth"],
    no_punc=True,
    speech_mode="yes",
    features_separator=None,
    opening_delim=None,
    closing_delim=None,
    do_tokenize=True,
)

print(f"Extracted {token_count} tokens")
print(text)
```

## Usage

All flags and their defaults can be viewed with:

```bash
aozora-corpus-generator --help
```

 You may pass multiple values to options like `--features` and `--author-title-csv` by repeating them or listing multiple arguments, for example:

```bash
aozora-corpus-generator --features orth lemma pos1 \
  --author-title-csv a.csv b.csv
```
