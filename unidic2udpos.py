#!/usr/bin/env python3

import os
import pathlib
import pprint
import re
import sys
from itertools import zip_longest

# Where we differ from GiNZA:
# ^形状詞-助動詞語幹,,AUX instead of ADJ
# ^空白,,SYM instead of SPACE

# Where we differ from BCCWJ:
# ^名詞-普通名詞-形状詞可能,,ADJ instead of NOUN
# ^接続詞,,SCONJ instead odf CCONJ
# ^補助記号-(句点|読点|括弧(閉|開)|一般),,PUNCT added 一般

udmap_list = """^形容詞-非自立可能,,ADJ
^形容詞,,ADJ
^連体詞,^[こそあど此其彼]の,DET
^連体詞,^[こそあど此其彼],PRON
^形状詞-一般,,ADJ
^形状詞-タリ,,ADJ
^形状詞-助動詞語幹,,AUX
^副詞,,ADV
^感動詞,,INTJ
^名詞-普通名詞-一般,,NOUN
^名詞-普通名詞-サ変可能,,NOUN
^名詞-普通名詞-形状詞可能,,ADJ
^名詞-普通名詞-副詞可能,,NOUN
^名詞-普通名詞-サ変形状詞可能,,NOUN
^名詞-普通名詞-助数詞可能,,NOUN
^名詞-数詞,,NUM
^名詞-助動詞語幹,,AUX
^名詞-固有名詞,,PROPN
^動詞-非自立可能,,VERB
^動詞,,VERB
^助詞-[格係副]助詞,,ADP
^助動詞,,AUX
^助詞-接続助詞,て,SCONJ
^接続詞,,SCONJ
^接続助?詞,,CCONJ
^連体詞,,ADJ
^助詞-準体助詞,,SCONJ
^助詞-[^格接副],,PART
^助詞-接続助詞,,CCONJ
^代名詞,,PRON
^補助記号-(句点|読点|括弧(閉|開)|一般),,PUNCT
^補助記号,,SYM
^記号,,SYM
^空白,,SYM
^接頭辞,,NOUN
^接尾辞,,PART
.,,X""".splitlines()

# TODO: ^助詞-[^格接副] not matching ^助詞-接続助詞 so manually added

ud_rules = [rule.split(',') for rule in udmap_list]
ud_rules = [[re.compile(rule[0]), re.compile(rule[1]) if rule[1] != '' else None, rule[2]]
            for rule in ud_rules]

pprint.pprint(ud_rules)


def convert_token(c, n):
    '''c = current_line, n = next_line'''
    orth, pos, lemma = c.split('\t')
    if not n or sentence_break(n):
        n_orth, n_pos, n_lemma = None, None, None
    else:
        n_orth, n_pos, n_lemma = n.split('\t')

    new_pos = None
    for (pos_re, lemma_re, ud_pos) in ud_rules:
        # https://github.com/megagonlabs/ginza/blob/develop/ginza/tag_map.py
        if lemma == '為る' and pos == '動詞-非自立可能':
            new_pos = 'AUX'
        elif pos == '名詞-普通名詞-サ変可能' and n_pos == '動詞-非自立可能':
            new_pos = 'VERB'
        elif pos == '名詞-普通名詞-サ変形状詞可能' and n_pos == '動詞-非自立可能':
            new_pos = 'VERB'
        elif pos == '名詞-普通名詞-サ変形状詞可能' and (n_pos == '助動詞' or n_pos.find('形状詞') >= 0):
            new_pos = 'ADJ'
        elif pos_re.match(pos):
            if lemma_re:
                if lemma_re.match(lemma):
                    new_pos = ud_pos
                    break
            else:
                new_pos = ud_pos
                break

    if new_pos is None:
        # We look for any gaps in lemma matching here:
        raise Exception((c, n))

    return (new_pos,) # (orth, pos, lemma, new_pos)


def sentence_break(s):
    if s == '<EOS>' or s == '<PGB>':
        return True
    else:
        return False


def partition_by_sentence(xs):
    n = len(xs)
    breaks = [i for i, x in enumerate(xs) if sentence_break(x)]
    for i in range(len(breaks)-1, 0, -1):
        if i > 0 and breaks[i] - breaks[i-1] == 1:
            del breaks[i]
    # breaks = [j for i, j in zip_longest(breaks, breaks[1:], fillvalue=0) if j - i != 1]
    i = 0
    for b in breaks:
        yield xs[i:b]
        i = b
    if i < n:
        yield xs[i:n]


def main(in_dir, out_dir):
    pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)

    for file in os.listdir(in_dir):
        if file.endswith('.txt'):
            with open(os.path.join(in_dir, file), encoding="utf-8") as fin, \
                 open(os.path.join(out_dir, file), 'w', encoding="utf-8") as fout:
                lines = fin.read().splitlines()
                for sentence in partition_by_sentence(lines):
                    for current_line, next_line in zip_longest(sentence, sentence[1:]):
                        if sentence_break(current_line):
                            fout.write(current_line + '\n')
                        else:
                            fout.write('\t'.join(convert_token(current_line, next_line)) + '\n')


if __name__ == '__main__':
    if len(sys.argv) < 3:
        raise Exception('First argument must be directory containing "orth\tpos\tlemma" type lines.\nYou must specify output directory as the second argument.')
    if not os.path.exists(sys.argv[1]):
        raise Exception('First argument must be directory containing orth\tpos\tlemma type lines.')
    if not sys.argv[2]:
        raise Exception('You must specify output directory as the second argument.')
    main(sys.argv[1], sys.argv[2])
