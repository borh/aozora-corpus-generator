#!/usr/bin/env python3

import re
import sys
import os
import pathlib
import pprint

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
^名詞-普通名詞-形状詞可能,,NOUN
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
^接続助?詞,,CCONJ
^連体詞,,ADJ
^助詞-準体助詞,,SCONJ
^助詞-[^格接副],,PART
^助詞-接続助詞,,PART
^代名詞,,PRON
^補助記号-(句点|読点|括弧(閉|開)),,PUNCT
^補助記号,,SYM
^記号,,SYM
^空白,,X
^接頭辞,,NOUN
^接尾辞,,PART""".splitlines()

# TODO: ^助詞-[^格接副] not matching ^助詞-接続助詞 so manually added

ud_rules = [rule.split(',') for rule in udmap_list]
ud_rules = [[re.compile(rule[0]), re.compile(rule[1]) if rule[1] != '' else None, rule[2]]
            for rule in ud_rules]

pprint.pprint(ud_rules)


def convert_line(line):
    orth, pos, lemma = line.split('\t')

    new_pos = None
    for (pos_re, lemma_re, ud_pos) in ud_rules:
        if pos_re.match(pos):
            if lemma_re:
                if lemma_re.match(lemma):
                    new_pos = ud_pos
                    break
            else:
                new_pos = ud_pos
                break

    if new_pos is None:
        raise Exception(line)

    return (orth, new_pos, lemma)


def main(in_dir, out_dir):
    pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)

    for file in os.listdir(in_dir):
        if file.endswith('.txt'):
            with open(os.path.join(in_dir, file)) as fin, \
                 open(os.path.join(out_dir, file), 'w') as fout:
                for line in fin:
                    line = line.rstrip('\n')
                    if line == '<EOS>' or line == '<PGB>':
                        fout.write(line + '\n')
                    else:
                        fout.write('\t'.join(convert_line(line)) + '\n')


if __name__ == '__main__':
    if len(sys.argv) < 3:
        raise Exception('First argument must be directory containing "orth\tpos\tlemma" type lines.\nYou must specify output directory as the second argument.')
    if not os.path.exists(sys.argv[1]):
        raise Exception('First argument must be directory containing orth\tpos\tlemma type lines.')
    if not sys.argv[2]:
        raise Exception('You must specify output directory as the second argument.')
    main(sys.argv[1], sys.argv[2])
