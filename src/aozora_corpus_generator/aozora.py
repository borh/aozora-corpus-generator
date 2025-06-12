# coding=utf-8
import csv
import importlib.resources
import io
import logging
import os
import pathlib
import re
import subprocess
import sys
import unicodedata
import urllib.request
import zipfile
from collections import (
    defaultdict,
)
from io import TextIOWrapper
from os.path import splitext
from typing import (
    DefaultDict,
    Dict,
    Generator,
    List,
    Optional,
    Tuple,
    cast,
)
from zipfile import ZipFile

import html5_parser as html
import jaconv
from fugashi import Tagger  # type: ignore

from aozora_corpus_generator.schemas import (
    BunruiInfo,
    CodeFrequencies,
    ConversionResult,
    CorpusMetadata,
    ProcessedText,
    Token,
    WorkMetadata,
)

# Lazy‐load the MeCab tagger only when tokenization is requested.
tagger: Optional[Tagger] = None


def ensure_unidic_dir(fallback_url: str) -> pathlib.Path:
    """
    Download & unpack UniDic-novel ZIP once into `<project_root>/unidic-novel/`
    if it does not already exist. Returns the extraction directory.
    """
    # go up three levels: .../src/aozora_corpus_generator/aozora.py → project root
    project_root = pathlib.Path(__file__).resolve().parent.parent.parent
    extract_dir = project_root / "unidic-novel"
    if not extract_dir.exists():
        log.info(f"Downloading UniDic-novel from {fallback_url}")
        resp = urllib.request.urlopen(fallback_url)
        with zipfile.ZipFile(io.BytesIO(resp.read())) as zf:
            zf.extractall(path=extract_dir)
    return extract_dir


def _get_tagger() -> Tagger:
    """
    Lazy import and return a MeCab/UniDic Tagger.
    Discovers the installed dictionary path by calling `mecab -D` and passing
    its parent directory to fugashi via `-d`.
    """
    global tagger
    if tagger is None:
        dict_dir: Optional[str] = None
        # try to discover the system dictionary via `mecab -D`
        try:
            out = subprocess.check_output(["mecab", "-D"], encoding="utf-8")
            for line in out.splitlines():
                if line.startswith("filename:"):
                    sys_dic = line.split(":", 1)[1].strip()
                    dict_dir = os.path.dirname(sys_dic)
                    break
        except (subprocess.CalledProcessError, FileNotFoundError):
            log.warning(
                "`mecab -D` failed, falling back to Fugashi default dictionary search."
            )

        if dict_dir:
            log.info(f"Using: -d {dict_dir} -r {dict_dir}/dicrc")
            tagger = Tagger(f"-d {dict_dir} -r {dict_dir}/dicrc")
        else:
            # if CLI pre‐downloaded the dict, use that
            explicit = os.environ.get("AOZORA_UNIDIC_DIR")
            log.info(f"Using: -d {explicit} -r {explicit}/dicrc")
            if explicit:
                tagger = Tagger(f"-d {explicit} -r {explicit}/dicrc")
            else:
                # otherwise if they gave us a URL, download now
                fallback_url = os.environ.get("AOZORA_UNIDIC_URL")
                if fallback_url:
                    extract_dir = ensure_unidic_dir(fallback_url)
                    log.info(f"Using: -d {extract_dir} -r {extract_dir}/dicrc")
                    tagger = Tagger(f"-d {extract_dir} -r {extract_dir}/dicrc")
                else:
                    # let Fugashi do its own lookup
                    tagger = Tagger()
    return tagger


log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
stdout_handler = logging.StreamHandler(sys.stderr)
stdout_handler.setLevel(logging.INFO)

_gaiji_map: Optional[Dict[str, str]] = None


def _get_gaiji_map() -> Dict[str, str]:
    """
    Lazy-load and cache the JIS X 0213 → Unicode mapping.
    """
    global _gaiji_map
    if _gaiji_map is None:
        _gaiji_map = make_jis_unicode_map()
    return _gaiji_map


def make_jis_unicode_map() -> Dict[str, str]:
    """
    Generates a translation dictionary between the men-ku-ten
    (i.e. '1-1-24') type of representation of characters in the JIS X
    0213 standard and Unicode. This format is used to represent most
    of the so-called 'gaiji' within Aozora Bunko, which refer to
    characters on the 3rd and 4th planes of the JIS X 0213
    standard. Note that this does not cover all 'gaiji' use, which
    includes references to Unicode itself or to a decription of the
    character as combination of two or more other chracters.
    Reference: http://www.aozora.gr.jp/annotation/external_character.html
    """
    d: Dict[str, str] = {}
    hex_to_code = dict(
        zip(
            [format(i, "X") for i in range(33, 33 + 95)],
            ["{0:0>2}".format(i) for i in range(1, 95)],
        )
    )

    # load the shipped mapping via importlib.resources
    with importlib.resources.open_text(
        "aozora_corpus_generator", "jisx0213-2004-std.txt"
    ) as f:
        for line in f:
            if line[0] == "#":
                continue

            jis_field, unicode_field = line.split("\t")[0:2]

            jis_standard, jis_code = jis_field.split("-")
            if jis_standard == "3":
                men = 1
            elif jis_standard == "4":
                men = 2

            ku = hex_to_code[jis_code[0:2]]
            ten = hex_to_code[jis_code[2:4]]

            unicode_point = unicode_field.replace("U+", "")
            if unicode_point == "":  # No mapping exists.
                continue
            elif len(unicode_point) > 6:  # 2 characters
                first, second = unicode_point.split("+")
                unicode_char = chr(int(first, 16)) + chr(int(second, 16))
            else:
                unicode_char = chr(int(unicode_point, 16))

            jis_string = "{}-{}-{}".format(men, ku, ten)

            d[jis_string] = unicode_char
    return d


# TODO ／″＼ and Unidic


def make_ndc_map() -> Dict[str, str]:
    # load the shipped NDC lookup via importlib.resources
    with importlib.resources.open_text(
        "aozora_corpus_generator", "ndc-3digits.tsv"
    ) as f:
        d: Dict[str, str] = {}
        for line in f:
            code, label = line.rstrip("\n").split("\t")
            d[code] = label
        return d


def make_bunrui_map(
    file_path: str = "wlsp2unidic/BunruiNo_LemmaID.txt",
) -> Dict[str, BunruiInfo]:
    with open(file_path) as f:
        d: Dict[str, BunruiInfo] = {}
        for line in f:
            bunrui_fields, lemma_id = line.rstrip("\n").split("\t")
            (
                article_number,
                _class_division_section_article_label,
                _article_number_paragraph_number_small_paragraph_number_word_number,
            ) = bunrui_fields.split(",")
            d[lemma_id] = {"article": article_number}
    return d


def normalize_japanese_text(s: str) -> str:
    """
    Normalizes to NFKC Unicode norm and converts all half-width
    alphanumerics and symbols to full-width. The latter step is needed
    for correct use with the UniDic dictionary used with MeCab.
    """
    return (
        jaconv.h2z(unicodedata.normalize("NFKC", s), kana=False, digit=True, ascii=True)
        .replace(".", "．")
        .replace(",", "，")
    )


q = 0


def split_sentence_ja(s: str) -> List[str]:
    """
    Splits Japanese text (paragraph) into sentences using common sentence
    delimiters. Some care is taken to prevent false positives, but more sophisticated
    (non-regex) logic would be required to implement a more robust solution.

    Examples:
    >>> split_sentence_ja("これはペンです。はい？いいえ！")
    ['これはペンです。', 'はい？', 'いいえ！']
    """
    global q
    # reset the global quote‐balance counter
    q = 0
    delimiters = r"[!\?。．！？…]+」?"

    # first, break adjacent quotes into separate lines
    s = s.replace("」「", "」\n「")

    # insert newlines at each sentence delimiter
    raw = re.sub(rf"({delimiters})", r"\1\n", s)
    # strip and drop any empty lines
    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]

    # post‐process: if a segment is *only* opening‐quote + ellipses, fuse it with the next segment
    merged: List[str] = []
    i = 0
    while i < len(lines):
        seg = lines[i]
        if re.fullmatch(r"[「…]+", seg) and i + 1 < len(lines):
            merged.append(seg + lines[i + 1])
            i += 2
        else:
            merged.append(seg)
            i += 1
    return merged


KATAKANA = set(
    list(
        "ァアィイゥウェエォオカガキギクグケゲコゴサザシジスズセゼソ"
        "ゾタダチヂッツヅテデトドナニヌネノハバパヒビピフブプヘベペ"
        "ホボポマミムメモャヤュユョヨラリルレロワヲンーヮヰヱヵヶヴ"
    )
)

HIRAGANA = set(
    list(
        "ぁあぃいぅうぇえぉおかがきぎくぐけげこごさざしじすず"
        "せぜそぞただちぢっつづてでとどなにぬねのはばぱひびぴ"
        "ふぶぷへべぺほぼぽまみむめもゃやゅゆょよらりるれろわ"
        "をんーゎゐゑゕゖゔ"
    )
)

KANJI_RX = re.compile(r"[\u4e00-\u9fff]")


def code_frequencies(text: str) -> Tuple[CodeFrequencies, Dict[str, int]]:
    """
    Examples:
    >>> cmap, types = code_frequencies("あアA")
    >>> cmap["hiragana"]
    1
    >>> cmap["katakana"]
    1
    >>> types["A"]
    1
    """
    cmap: CodeFrequencies = {
        "katakana": 0,
        "hiragana": 0,
        "kanji": 0,
        "other": 0,
    }

    unigram_types: Dict[str, int] = defaultdict(int)

    for c in text:
        unigram_types[c] += 1
        if c in KATAKANA:
            cmap["katakana"] += 1
        elif c in HIRAGANA:
            cmap["hiragana"] += 1
        elif KANJI_RX.match(c):
            cmap["kanji"] += 1
        else:
            cmap["other"] += 1

    return cmap, unigram_types


def is_katakana_sentence(text: str, tokens: List[Token]) -> bool:
    """
    Identify if sentence is made up of katakana (ie. no hiragana)
    characters. In such a case, it should be converted to hiragana,
    processed, and then the original orth should be substituted back
    in.

    Note: This identification is an imperfect heuristic and should
    be replaced.
    """

    cmap, unigram_types = code_frequencies(text)

    bigram_types: Dict[str, int] = defaultdict(int)
    for bigram in map(lambda a, b: "{}{}".format(a, b), text, text[1:]):
        bigram_types[bigram] += 1

    total_chars = cmap["katakana"] + cmap["hiragana"] + cmap["kanji"] + cmap["other"]
    katakana_ratio = cmap["katakana"] / total_chars

    if cmap["hiragana"] == 0 and katakana_ratio > 0.5:
        oov_count = sum(1 for token in tokens if token["is_unk"])
        proper_noun_chars = sum(
            len(token["orth"]) for token in tokens if token["pos2"] == "固有名詞"
        )

        if oov_count == 0 and proper_noun_chars / len(text) > 0.3:
            return False

        if (
            oov_count / len(tokens) > 0.2
            or len(text) < 8
            or (len(text) < 10 and re.search(r"ッ?.?[？！]」$", text[-3:]))
            or (len(text) < 100 and len(unigram_types) / len(text) < 0.5)
            or max(bigram_types.values()) / len(text) > 0.5
            or len(re.findall(r"(.)\1+", text)) / len(text) > 0.1
            or len(re.findall(r"(..)ッ?\1", text)) / len(text) > 0.1
        ):
            return False
        return True
    else:
        return False


def sentence_to_tokens(sentence: str, is_katakana: bool = False) -> List[Token]:
    """
    Parses one sentence into tokens using MeCab. Assumes UniDic CWJ
    2.3.0 version of the dictionary is set as default. If is_katakana
    is set, then will convert hiragana to katakana before passing the
    string to MeCab and then finally reverting the change to the
    surface form.
    """
    tokens: List[Token] = []

    if is_katakana:
        sentence = jaconv.kata2hira(sentence)
    # now use lazy tagger
    tagger_inst = _get_tagger()
    for parsed_token in tagger_inst(sentence):
        try:
            token_dict = dict(parsed_token.feature)
        except Exception:
            token_dict = dict(parsed_token.feature._asdict())

        # We need to map the surface string to OOV tokens:
        orth_val = token_dict.get("orth") or parsed_token.surface
        orth_base_val = token_dict.get("orthBase") or orth_val
        lemma_val = token_dict.get("lemma") or orth_val
        token: Token = {
            "surface": parsed_token.surface,
            "is_unk": parsed_token.is_unk,
            "pos": parsed_token.pos,
            "orth": orth_val,
            "orthBase": orth_base_val,
            "lemma": lemma_val,
            "pos1": token_dict.get("pos1", ""),
            "pos2": token_dict.get("pos2", ""),
            "pos3": token_dict.get("pos3"),
            "pos4": token_dict.get("pos4"),
            "pos5": token_dict.get("pos5"),
            "pos6": token_dict.get("pos6"),
            "cType": token_dict.get("cType"),
            "cForm": token_dict.get("cForm"),
            "lForm": token_dict.get("lForm"),
            "lemma_id": token_dict.get("lemma_id"),
        }

        if token["is_unk"]:  # OOV
            if is_katakana:
                token["orth"] = jaconv.hira2kata(token["orth"])
            token["orthBase"] = token["orth"]
            token["lemma"] = token["orth"]
        else:
            if is_katakana:
                token["orth"] = jaconv.hira2kata(token["orth"])

        tokens.append(token)
    return tokens


# Quotations may transcend sentence boundaries, so we only optionally match closing quotations.
SPEECH_RX = re.compile(r"(「「[^「」]+」?)")


## FIXME: Sentences beginning with  ―― in Kobayashi_T_Koujousai.txt are ds
def text_to_tokens(
    paragraph: str, speech_mode: str = "yes"
) -> Generator[List[Token], None, None]:
    """
    Returns a sequence of sentences, each comprising a sequence of
    tokens. Must be subsumed into non-lazy collection.  Will re-parse
    the sentence if it detects it as written in kanji-katakana-majiri
    form. speech_mode controls the presence or absence of direct
    speech.
    """

    # unclosed_quotation = False
    # We copy q from global state before splitting, as we need to replicate the     sentence to q balance mappings.
    ql = q
    sentences = split_sentence_ja(paragraph)
    for sentence in sentences:
        if speech_mode != "yes":  # Speech detection is on.
            speech_matches = SPEECH_RX.findall(sentence)
            opening_q_count = sentence.count("「")
            closing_q_count = sentence.count("」")

            ql += opening_q_count - closing_q_count

            if ql > 0 and closing_q_count == 0:
                # While in speech paragraph, all text is speech.
                speech_matches = [sentence]
            elif closing_q_count > opening_q_count:
                # Unless this sentence contains the closing quotation mark.
                speech_matches = re.findall(r"([^」]+」)", sentence)
            elif ql < 0:
                pass
                # log.error(f'{ql} is not a valid balance count:\n {sentence}     \n\nin\n\n {paragraph}')

            # if unclosed_quotation:
            #     if closing_q_count == 0:
            #         speech_matches = [sentence]
            #     elif closing_q_count > opening_q_count:
            #         speech_matches = re.findall(r'([^」]+」)', sentence)
            #         unclosed_quotation = False
            #     else:
            #         print(sentence)
            # elif closing_q_count < opening_q_count:
            #     unclosed_quotation = True

            if speech_matches:  # or unclosed_quotation:
                s = sentence
                if speech_mode == "speech":  # Only add speech to s.
                    s = ""
                for maybe_speech in speech_matches:
                    cmap, _ = code_frequencies(re.sub(r"[「」]", "", maybe_speech))
                    total_types = (
                        cmap["katakana"]
                        + cmap["hiragana"]
                        + cmap["kanji"]
                        + cmap["other"]
                    )
                    if cmap["kanji"] == total_types and maybe_speech.startswith("「"):
                        # False positive. (crude detection heuristic)
                        # Ideally test for minimal sentence...
                        # We do not reject if we are already in a speech paragraph.
                        print("Rejecting:", maybe_speech)
                    elif speech_mode == "no":
                        # Delete speech from sentence. This should probably not be     used, as it may affect the syntactic
                        # structure.
                        s = re.sub(r"「?{}」?".format(maybe_speech), "", s)
                    elif speech_mode == "narrative":
                        s = ""
                        break
                    elif speech_mode == "speech":
                        s += maybe_speech
                    else:
                        raise Exception("Speech mode invalid", speech_mode)
                sentence = s
            elif speech_mode == "yes":
                # Ignore sentence if not speech.
                sentence = ""

        if not sentence:
            continue

        tokens = sentence_to_tokens(sentence)
        if len(tokens) == 0:
            continue

        is_katakana = is_katakana_sentence(sentence, tokens)
        if is_katakana:
            tokens = sentence_to_tokens(sentence, is_katakana)

        yield tokens


PUNC_RX = re.compile(r"^((補助)?記号|空白)$")
NUMBER_RX = re.compile(r"^[\d０-９一-九]+$")


def wakati(
    text: str, no_punc: bool = True, speech_mode: str = "yes"
) -> Generator[List[str], None, None]:
    """
    Returns a sequence of sentences comprised of whitespace separated tokens.
    """
    for sentence in text_to_tokens(text, speech_mode):
        if no_punc:
            yield [
                token["orth"]
                for token in sentence
                if not PUNC_RX.match(token["pos1"])
                and not (token["pos2"] == "数詞" and NUMBER_RX.match(token["orth"]))
            ]
        else:
            yield [token["orth"] for token in sentence]


# import pprint
def tokenize(
    paragraph_text: str,
    features: List[str],
    no_punc: bool = True,
    speech_mode: str = "yes",
    features_separator: Optional[str] = None,
    opening_delim: Optional[str] = None,
    closing_delim: Optional[str] = None,
) -> Generator[List[str], None, None]:
    """
    Returns a sequence of sentences comprised of whitespace separated
    tokens. Supports encoding tokens with other POS or morphological
    annotations.
    """
    first_feature = features[0]
    rest_features = features[1:]

    # Transform tab to real tab to deal with terminal passthrough issue.
    if opening_delim == "tab":
        opening_delim = "\t"
    if closing_delim == "tab":
        closing_delim = "\t"

    if len(features) == 1:
        opening_delim, closing_delim = "", ""
    else:
        if not opening_delim:
            opening_delim, closing_delim = "/", ""
        if not closing_delim:
            closing_delim = ""

    if not features_separator:
        features_separator = ","
    elif features_separator == "tab":
        features_separator = "\t"

    for sentence in text_to_tokens(paragraph_text, speech_mode):
        if no_punc:
            tokens = [
                str(
                    token[first_feature]  # type: ignore[literal-required]
                    + opening_delim
                    + features_separator.join(
                        token[feature]  # type: ignore[literal-required]
                        for feature in rest_features
                    )
                    + closing_delim
                ).replace("\n", "")
                for token in sentence
                if not PUNC_RX.match(token["pos1"])
                and not (token["pos2"] == "数詞" and NUMBER_RX.match(token["orth"]))
            ]
        else:
            tokens = [
                "{}{}{}{}".format(
                    token[first_feature],  # type: ignore[literal-required]
                    opening_delim,
                    features_separator.join(
                        token[feature]  # type: ignore[literal-required]
                        for feature in rest_features
                    ),
                    closing_delim,
                ).replace("\n", "")
                for token in sentence
            ]

        if tokens:
            yield tokens


def romanize(s: str) -> str:
    """
    Convert Katakana/Hiragana → ASCII.

    Examples:
    >>> romanize("カタカナ")
    'katakana'

    >>> romanize("こんにちは")
    'konnichiha'
    """
    return re.sub(
        r"_+",
        "_",
        re.sub(
            r"[^a-zA-Z]",
            "_",
            jaconv.kana2alphabet(jaconv.kata2hira(s.replace("ゔ", "v"))),
        ),
    )


def read_aozora_bunko_list(
    path: str, ndc_tr: Dict[str, str]
) -> DefaultDict[str, Dict[str, WorkMetadata]]:
    """
    Reads in the list_person_all_extended_utf8.csv of Aozora Bunko and
    constructs a nested dictionary keyed on author and title. This is
    then used identify the correct path to the file as well as give
    more metadata.
    """
    d: DefaultDict[str, Dict[str, WorkMetadata]] = defaultdict(dict)
    url_rx = re.compile(r"https://www\.aozora\.gr\.jp/cards/(\d+)/(.+)")
    with ZipFile(path) as z:
        with z.open("list_person_all_extended_utf8.csv", "r") as f:
            for row in csv.DictReader(TextIOWrapper(f)):
                # Some works have versions in both new- and old-style
                # kana. As we are only interested in the new-style
                # version, we skip the old one while keeping only
                # old-style works.
                if row["文字遣い種別"] != "新字新仮名":
                    log.debug(f"Skipping processing of old-syle kana work: {row}")
                    continue

                # Use the lower value from 底本初版発行年1 and 初出:
                year = ""

                year_rx = re.compile(r"(\d{4})（.+）年\s?(\d{1,2})月((\d{1,2})日)?")

                year_matches = year_rx.match(row["底本初版発行年1"])
                if year_matches and year_matches.groups():
                    year = year_matches.groups()[0]

                year_alternate_matches = year_rx.search(row["初出"])
                if year_alternate_matches and year_alternate_matches.groups():
                    alt_year = year_alternate_matches.groups()[0]
                    if year == "":
                        year = alt_year
                    elif int(alt_year) < int(year):
                        year = alt_year

                # Sanity check for year:
                year_death = re.search(r"\d{4}", row["没年月日"])
                if (
                    year_death
                    and year_death.groups()
                    and int(year_death.group(0)) < int(year)
                ):
                    year = "<" + year_death.group(
                        0
                    )  # Specify upper bound as last resort.

                author_ja = row["姓"] + row["名"]
                author_en = row["名ローマ字"] + " " + row["姓ローマ字"]
                title = row["作品名"]
                title_ja = title
                title_en = jaconv.kana2alphabet(
                    jaconv.kata2hira(row["作品名読み"])
                ).title()
                subtitle = row["副題"]
                if subtitle != "":
                    title_ja += ": " + subtitle
                    title_en += ": " + romanize(row["副題読み"]).title()

                match = url_rx.match(row["XHTML/HTMLファイルURL"])
                if not match:
                    log.debug(f"Missing XHTML/HTML file for record {row}, skipping...")
                    continue
                card_id = match.group(1)
                file_path = match.group(2)

                ndc = row["分類番号"].replace("NDC ", "").replace("K", "")

                if len(ndc) > 3:
                    ndcs = ndc.split()
                    ndc = "/".join(ndc_tr[n] for n in ndcs)
                elif not ndc:
                    ndc = ""
                else:
                    ndc = ndc_tr[ndc]

                if "K" in row["分類番号"]:
                    ndc += " (児童書)"

                if title in d[author_ja]:
                    # Remove translations.
                    d[author_ja].pop(title, None)
                    if len(d[author_ja]) == 0:
                        d.pop(author_ja, None)
                else:
                    d[author_ja][title] = {
                        "author_ja": author_ja,
                        "author": author_en,
                        "author_year": f"{row['生年月日']}--{row['没年月日']}",
                        "title_ja": title_ja,
                        "title": title_en,
                        "year": year,
                        "ndc": ndc,
                        "file_path": f"aozorabunko/cards/{card_id}/{file_path}",
                        "file_name": "{}_{}_{}".format(  # TODO Do we need to shorten these?
                            row["姓ローマ字"],
                            row["名ローマ字"][0:1],
                            romanize(row["作品名読み"][0:7]).title(),
                        ),
                    }
    return d


def read_author_title_list(
    aozora_db: DefaultDict[str, Dict[str, WorkMetadata]], path: str
) -> Tuple[List[Tuple[str, str, str]], List[CorpusMetadata]]:
    """
    Reads in the author title table that is used to extract a subset
    of author-title pairs from Aozora Bunko. The CSV file must contain
    the columns 'author' and 'title'. Output is a list of corpus files
    and a database containing metadata.

    The reader supports an optional '*' value for the title field. If
    it encounters one, it will match on all the works of the
    author. To extract all texts from Aozora Bunko, see the `--all`
    flag.
    """
    corpus_files: List[Tuple[str, str, str]] = []
    db: List[CorpusMetadata] = []
    # base directory for any non-Aozora files
    csv_dir = pathlib.Path(path).parent
    with open(path, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            if row["corpus"] == "Aozora Bunko":
                auth = re.sub(r"\s", "", row["author"])
                try:
                    works = aozora_db[auth]
                    if row["title"] == "*":
                        for title, m in works.items():
                            corpus_files.append(
                                (row["corpus"], m["file_name"], m["file_path"])
                            )
                            db.append(
                                {
                                    "corpus": row["corpus"],
                                    "corpus_id": m["file_path"],
                                    "author": auth,
                                    "author_ja": m.get("author_ja", ""),
                                    "title": title,
                                    "title_ja": m.get("title_ja", ""),
                                    "brow": "",
                                    "genre": "",
                                    "narrative_perspective": "",
                                    "comments": "",
                                    "token_count": None,
                                    "author_year": m.get("author_year", ""),
                                    "year": m.get("year", ""),
                                    "ndc": m.get("ndc", ""),
                                }
                            )
                    else:
                        m = works[row["title"]]
                        corpus_files.append(
                            (row["corpus"], m["file_name"], m["file_path"])
                        )
                        db.append(
                            {
                                "corpus": row["corpus"],
                                "corpus_id": m["file_path"],
                                "author": auth,
                                "author_ja": m.get("author_ja", ""),
                                "title": row["title"],
                                "title_ja": m.get("title_ja", ""),
                                "brow": "",
                                "genre": "",
                                "narrative_perspective": "",
                                "comments": "",
                                "token_count": None,
                                "author_year": m.get("author_year", ""),
                                "year": m.get("year", ""),
                                "ndc": m.get("ndc", ""),
                            }
                        )
                except KeyError:
                    log.warning(
                        f"{auth}/{row['title']} not in Aozora Bunko DB. Skipping..."
                    )
            else:
                # 1) prefer an explicit filename column
                filename = row.get("filename")
                # 2) if missing, build author_title.html
                if not filename:
                    author_base = row["author"].replace(" ", "_")
                    title_base = row["title"].replace(" ", "_")
                    filename = f"{author_base}_{title_base}.html"

                # 3) if no ext, try .html then .xhtml
                base, ext = splitext(filename)
                # decide directory: use corpus subdir if it exists, otherwise use csv_dir
                subdir = csv_dir / row["corpus"]
                file_dir = subdir if subdir.is_dir() else csv_dir
                if not ext:
                    if (file_dir / f"{base}.html").exists():
                        filename = f"{base}.html"
                    elif (file_dir / f"{base}.xhtml").exists():
                        filename = f"{base}.xhtml"

                file_path = str(file_dir / filename)
                file_basename = splitext(filename)[0]

                corpus_files.append((row["corpus"], file_basename, file_path))
                row["filename"] = filename
                row["corpus_id"] = file_path
                # fill in all metadata fields so write_metadata_file won’t KeyError
                row["brow"] = row.get("brow", "")
                row["genre"] = row.get("genre", "")
                row["narrative_perspective"] = row.get("narrative_perspective", "")
                row["comments"] = row.get("comments", "")
                # also supply the “ja”‐only fields as blanks
                row["author_ja"] = row.get("author_ja", "")
                row["title_ja"] = row.get("title_ja", "")
                row["author_year"] = row.get("author_year", "")
                row["year"] = row.get("year", "")
                row["ndc"] = row.get("ndc", "")
                row["token_count"] = None

                db.append(cast(CorpusMetadata, row))

    return corpus_files, db


def remove_from(s: str, pattern: str) -> str:
    """
    Truncate string at the first match of the given pattern.

    Examples:
    >>> remove_from("foo123bar", r"\\d+")
    'foo'
    """
    rx = re.compile(pattern, re.M)
    maybe_match = rx.search(s)
    if maybe_match:
        log.warning("Removing: {}".format(s[maybe_match.start() :]))
        return s[0 : maybe_match.start()]
    else:
        return s


def read_aozora_bunko_xml(
    path: str,
    features: List[str],
    no_punc: bool,
    speech_mode: str,
    features_separator: Optional[str],
    opening_delim: Optional[str],
    closing_delim: Optional[str],
    do_tokenize: bool = True,
    gaiji_tr: Optional[Dict[str, str]] = None,
) -> ProcessedText:
    """
    Reads an Aozora Bunko XHTML/HTML file and converts it into plain
    text. All comments and ruby are removed, and gaiji are replaced
    with Unicode equivalents.
    Reference:
    -   http://www.aozora.gr.jp/annotation/
    -   http://www.aozora.gr.jp/annotation/henkoten.html
    """

    # if caller did not supply a mapping, load the default
    if gaiji_tr is None:
        gaiji_tr = _get_gaiji_map()
    with open(path, "rb") as f:
        doc = html.parse(
            f.read(),
            maybe_xhtml=False,
            fallback_encoding="shift_jis",
            return_root=False,
        )
    body = doc.xpath(".//div[@class='main_text']")

    if len(body) == 0:
        log.warning(
            "Error extracting main_text from file {}, trying workaround...".format(path)
        )
        body = doc.xpath(".//body")
        if len(body) == 0:
            log.critical("Error extracting text from file {} by any means".format(path))
            return {"text": "", "paragraphs": [], "token_count": 0}
        else:
            body = body[0]
    else:
        body = body[0]

    # Remove ruby and notes:
    for e in body.xpath(
        """
      .//span[@class='notes']
    | .//rp
    | .//rt
    | .//sub
    | .//div[@class='bibliographical_information']
    | .//div[@class='notation_notes']
    | .//div[@class='after_text']
    """
    ):
        parent = e.getparent()
        assert parent is not None
        if e.tail:
            previous = e.getprevious()
            if previous is None:
                parent.text = (parent.text or "") + e.tail
            else:
                previous.tail = (previous.tail or "") + e.tail
        parent.remove(e)

    # Convert gaiji img tags to Unicode characters:
    for gaiji_el in body.xpath(".//img[@class='gaiji']"):
        src = gaiji_el.get("src") or ""
        gmatch = re.match(r".+gaiji/\d+-\d+/(\d-\d+-\d+)\.png", src)
        if not gmatch:
            continue
        menkuten_code = gmatch.group(1)
        uni = gaiji_tr.get(menkuten_code)
        if not uni:
            continue
        gaiji_el.text = uni
        log.debug(f"Replacing JIS X {menkuten_code} with Unicode '{uni}'")

    text = re.sub(
        r"[\r\n]+", "\n", "".join(body.itertext()).strip(), flags=re.MULTILINE
    )
    text = remove_from(
        text, r"^[　【]?(底本：|訳者あとがき|この翻訳は|この作品.*翻訳|この翻訳.*全訳)"
    )

    # if user only wants plain text, bail out here
    if not do_tokenize:
        return {"text": text, "paragraphs": [], "token_count": 0}

    global q
    q = 0

    paragraphs = [
        list(
            tokenize(
                paragraph,
                features,
                no_punc=no_punc,
                speech_mode=speech_mode,
                features_separator=features_separator,
                opening_delim=opening_delim,
                closing_delim=closing_delim,
            )
        )
        for paragraph in text.splitlines()
    ]

    if q != 0:
        log.error(
            f"q {q} not 0 in {path}, resetting... {text.count('「')} <> {text.count('」')}"
        )
        q = 0

    token_count = sum(
        len(sentence) for paragraph in paragraphs for sentence in paragraph
    )

    return {"text": text, "paragraphs": paragraphs, "token_count": token_count}


def write_corpus_file(
    text: str, paragraphs: List[List[List[str]]], file_name: str, prefix: str
) -> None:
    """
    Given a sequence of paragraphs and path to output, writes plain
    and tokenized versions of the paragraphs.
    """
    # always write output files as UTF-8
    with (
        open(
            f"{prefix}/Tokenized/{file_name}.txt", "w", encoding="utf-8"
        ) as f_tokenized,
        open(f"{prefix}/Plain/{file_name}.txt", "w", encoding="utf-8") as f_plain,
    ):
        # f_plain.write(text)
        f_plain.write(
            "\n\n".join(
                "\n".join("".join(sentence) for sentence in paragraph)
                for paragraph in paragraphs
            )
        )
        f_tokenized.write(
            "\n\n".join(
                "\n".join(" ".join(sentence) for sentence in paragraph)
                for paragraph in paragraphs
            )
        )
        # for paragraph in paragraphs:
        # f_tokenized.write('<PGB>\n'.join('\n'.join(sentence) + '\n<EOS>\n'
        #                                  for sentence in paragraph))


def convert_corpus_file(
    corpus: str,
    file_name: str,
    file_path: str,
    prefix: str,
    features: List[str] = ["orth"],
    no_punc: bool = True,
    speech_mode: str = "yes",
    min_tokens: int = 0,
    features_separator: Optional[str] = None,
    opening_delim: Optional[str] = None,
    closing_delim: Optional[str] = None,
    gaiji_tr: Optional[Dict[str, str]] = None,
) -> ConversionResult:
    """
    Helper function that reads in html and writes a plain/tokenized
    version in one step. Needed for concurrent.futures.
    """
    # for Aozora Bunko, load gaiji map on‐demand if not supplied
    if corpus == "Aozora Bunko" and gaiji_tr is None:
        gaiji_tr = _get_gaiji_map()

    # non‐Aozora: if it's HTML/XHTML, re‐use our XML→TEXT logic,
    # otherwise treat it as raw UTF‐8 text.
    if corpus != "Aozora Bunko":
        ext = pathlib.Path(file_path).suffix.lower()
        if ext in (".html", ".xhtml"):
            # parse exactly like Aozora HTML
            processed = read_aozora_bunko_xml(
                file_path,
                features,
                no_punc,
                speech_mode,
                features_separator,
                opening_delim,
                closing_delim,
                do_tokenize=True,
                gaiji_tr=gaiji_tr,
            )
            text = processed["text"]
            paragraphs = processed["paragraphs"]
            token_count = processed["token_count"]
        else:
            # plain UTF‐8 text
            with open(file_path, encoding="utf-8") as f:
                text = f.read()
            # use tokenize() so paragraphs: List[List[List[str]]], not Token
            paragraphs = [
                list(
                    tokenize(
                        p,
                        features,
                        no_punc=no_punc,
                        speech_mode=speech_mode,
                        features_separator=features_separator,
                        opening_delim=opening_delim,
                        closing_delim=closing_delim,
                    )
                )
                for p in text.splitlines()
            ]
            token_count = sum(len(s) for para in paragraphs for s in para)
        reject = True if (min_tokens and token_count < min_tokens) else False
        if not reject:
            write_corpus_file(text, paragraphs, file_name, prefix)
        return {
            "file_name": file_name,
            "file_path": file_path,
            "prefix": prefix,
            "token_count": token_count,
            "rejected": reject,
        }
    else:
        try:
            aozora_result: ProcessedText = read_aozora_bunko_xml(
                file_path,
                features,
                no_punc,
                speech_mode,
                features_separator,
                opening_delim,
                closing_delim,
                gaiji_tr=gaiji_tr,
            )
            text = aozora_result["text"]
            paragraphs = aozora_result["paragraphs"]
            token_count = aozora_result["token_count"]
        except UnicodeDecodeError as e:
            text, paragraphs, token_count = "", [], 0
            log.warning(f"Decoding of {file_path} failed with {e}")
    reject = True if (min_tokens and token_count < min_tokens) else False
    if not reject:
        write_corpus_file(text, paragraphs, file_name, prefix)
    return {
        "file_name": file_name,
        "file_path": file_path,
        "prefix": prefix,
        "token_count": token_count,
        "rejected": reject,
    }


def write_metadata_file(
    files: List[Tuple[str, str, str]],
    metadata: List[CorpusMetadata],
    aozora_db: DefaultDict[str, Dict[str, WorkMetadata]],
    prefix: str,
) -> None:
    """
    Writes metadata of processed author-title pairs for further
    analysis.
    """

    # pprint.pprint(aozora_db)

    metadata_fn = "{}/groups.csv".format(prefix)
    with open(metadata_fn, "w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(
            [
                "filename",
                "brow",
                "language",
                "corpus",
                "corpus_id",
                "author_ja",
                "title_ja",
                "author",
                "title",
                "author_year",
                "year",
                "token_count",
                "ndc",
                "genre",
                "narrative_perspective",
                "comments",
            ]
        )
        for (corpus, file_name, _), d in zip(files, metadata):
            if corpus != "Aozora Bunko":
                writer.writerow(
                    [
                        file_name + ".txt",
                        d["brow"],
                        "ja",
                        corpus,
                        d["corpus_id"],
                        d["author_ja"],
                        d["title_ja"],
                        d["author"],
                        d["title"],
                        d["author_year"],
                        d["year"],
                        d["token_count"],
                        d["ndc"],
                        d["genre"],
                        d["narrative_perspective"],
                        d["comments"],
                    ]
                )
            else:
                try:
                    m = aozora_db[d["author"]][d["title"]]
                    writer.writerow(
                        [
                            file_name + ".txt",
                            d["brow"],
                            "ja",
                            corpus,
                            d["corpus_id"],
                            m["author_ja"],
                            m["title_ja"],
                            m["author"],
                            m["title"],
                            m["author_year"],
                            m["year"],
                            d["token_count"],
                            m["ndc"],
                            d["genre"],
                            d["narrative_perspective"],
                            d["comments"],
                        ]
                    )
                except KeyError:
                    log.critical(f'Missing keys for {file_name} in d="{d}"')
        log.info("Wrote metadata to {}".format(metadata_fn))
