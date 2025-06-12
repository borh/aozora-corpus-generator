from typing import List, Optional, TypedDict


class Token(TypedDict):
    surface: str
    is_unk: bool
    pos: str
    orth: str
    orthBase: str
    lemma: str
    pos1: str
    pos2: str
    pos3: Optional[str]
    pos4: Optional[str]
    pos5: Optional[str]
    pos6: Optional[str]
    cType: Optional[str]
    cForm: Optional[str]
    lForm: Optional[str]
    lemma_id: Optional[str]


class WorkMetadata(TypedDict):
    author_ja: str
    author: str
    author_year: str
    title_ja: str
    title: str
    year: str
    ndc: str
    file_path: str
    file_name: str


class CorpusMetadata(TypedDict):
    corpus: str
    corpus_id: str
    author: str
    author_ja: Optional[str]
    title: str
    title_ja: Optional[str]
    brow: str
    genre: str
    narrative_perspective: str
    comments: str
    token_count: Optional[int]
    author_year: Optional[str]
    year: Optional[str]
    ndc: Optional[str]


class BunruiInfo(TypedDict):
    article: str


class CodeFrequencies(TypedDict):
    katakana: int
    hiragana: int
    kanji: int
    other: int


class ProcessedText(TypedDict):
    text: str
    paragraphs: List[List[List[str]]]
    token_count: int


class ConversionResult(TypedDict):
    file_name: str
    file_path: str
    prefix: str
    token_count: int
    rejected: bool
