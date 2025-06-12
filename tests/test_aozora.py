# coding=utf-8
from aozora_corpus_generator.aozora import (
    code_frequencies,
    normalize_japanese_text,
    remove_from,
    romanize,
    split_sentence_ja,
)


def test_sentence_splitting():
    assert split_sentence_ja(
        "「え？　どうだか……」「……全くです……知らないんですから……罪ですね」「まさか……」「バッタを……本当ですよ」"
    ) == [
        "「え？",
        "どうだか……」",
        "「……全くです……",
        "知らないんですから……",
        "罪ですね」",
        "「まさか……」",
        "「バッタを……",
        "本当ですよ」",
    ]


def test_code_frequencies_and_types():
    cmap, types = code_frequencies("漢Aあア.")
    # one kanji, one Latin, one hiragana, one katakana, one punctuation
    assert cmap["kanji"] == 1
    assert cmap["hiragana"] == 1
    assert cmap["katakana"] == 1
    # Latin and punctuation are both classified as "other"
    assert cmap["other"] == 2
    # unigram_types should count each character
    assert types["漢"] == 1
    assert types["A"] == 1
    assert types["."] == 1


def test_normalize_japanese_text_converts_to_fullwidth():
    s = "abc.DEF,123"
    out = normalize_japanese_text(s)
    # ascii letters, period, comma, and digits → fullwidth
    assert "ａｂｃ．ＤＥＦ，１２３" in out


def test_remove_from_truncates_match_and_preserves_no_match():
    # no digits → string unchanged
    assert remove_from("foobar", r"\d+") == "foobar"
    # digits present → truncate before first digit
    assert remove_from("foo123bar", r"\d+") == "foo"


def test_romanize_examples_match_doctests():
    assert romanize("カタカナ") == "katakana"
    assert romanize("こんにちは") == "konnichiha"
