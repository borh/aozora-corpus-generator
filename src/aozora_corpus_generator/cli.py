import argparse
import concurrent.futures
import logging
import os
import pathlib
import sys
import textwrap
from typing import Any, Dict

from aozora_corpus_generator.aozora import (
    convert_corpus_file,
    ensure_unidic_dir,
    log,
    make_ndc_map,
    read_aozora_bunko_list,
    read_aozora_bunko_xml,
    read_author_title_list,
    stdout_handler,
    write_metadata_file,
)
from aozora_corpus_generator.schemas import ConversionResult, CorpusMetadata


def parse_args() -> Dict[str, Any]:
    """Parses and returns program arguments as a dictionary."""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent(
            """aozora-corpus-generator extracts given author and book pairs from Aozora Bunko and formats them into (optionally tokenized) plain text files."""
        ),
        epilog=textwrap.dedent("""Example usage:
uv run aozora-corpus-generator --features 'orth' --author-title-csv 'author-title.csv' --out 'Corpora/Japanese' --parallel"""),
    )

    # if user typed no flags at all, show help and exit
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)
    parser.add_argument(
        "--features",
        nargs="+",
        help="specify which features should be extracted from morphemes (default='orth')",
        default=["orth"],
        required=False,
    )
    parser.add_argument(
        "--features-opening-delim",
        help="specify opening char to use when outputting multiple features",
        required=False,
    )
    parser.add_argument(
        "--features-closing-delim",
        help="specify closing char to use when outputting multiple features",
        required=False,
    )
    parser.add_argument(
        "--features-separator",
        help="specify separating char to use when outputting multiple features",
        required=False,
    )
    parser.add_argument(
        "--author-title-csv",
        nargs="+",
        help="one or more UTF-8 formatted CSV input file(s) (default='author-title.csv')",
        default=["author-title.csv"],
        required=False,
    )
    parser.add_argument(
        "--aozora-bunko-repository",
        help="path to the aozorabunko git repository (default='aozorabunko/index_pages/list_person_all_extended_utf8.zip')",
        default="aozorabunko/index_pages/list_person_all_extended_utf8.zip",
        required=False,
    )
    parser.add_argument(
        "--out",
        help="output (plain, tokenized) files into given output directory (default=Corpora)",
        default="Corpora",
        required=False,
    )
    parser.add_argument(
        "--all",
        help="specify if all Aozora Bunko texts should be extracted, ignoring the author-title.csv (default=False)",
        action="store_true",
        default=False,
        required=False,
    )
    parser.add_argument(
        "--min-tokens",
        help="specify minimum token count to filter files by (default=30000)",
        default=30000,
        type=int,
        required=False,
    )
    parser.add_argument(
        "--no-punc",
        help="specify if punctuation should be discarded from tokenized version (default=False)",
        action="store_true",
        default=False,
        required=False,
    )
    parser.add_argument(
        "--speech-mode",
        help='specify if direct speech should be included in tokenized version (default="yes"); '
        'a value of "narrative" will remove all sentences containing direct speech, and, '
        'conversely, a value of "speech" will only keep direct speech ("no" will remove only '
        "the speech from the sentence).",
        default="yes",
        choices=["yes", "no", "narrative", "speech"],
        required=False,
    )
    parser.add_argument(
        "--incremental",  # TODO
        help="do not overwrite existing corpus files (default=False)",
        action="store_true",
        default=False,
        required=False,
    )
    parser.add_argument(
        "--parallel",
        help="specify if processing should be done in parallel (default=True)",
        action="store_true",
        default=False,
        required=False,
    )
    parser.add_argument(
        "--verbose",
        help="turns on verbose logging (default=False)",
        action="store_true",
        default=False,
        required=False,
    )
    parser.add_argument(
        "--plaintext-only",
        help="generate only plain‐text files (skip tokenization & MeCab)",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--xhtml-file",
        help="process a single Aozora Bunko XHTML/HTML file and output its plain text to stdout",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--split-sentences",
        help="in single-file mode, split paragraphs into sentences like the Plain export",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--tokenized-only",
        help="in single-file mode, output tokenized sentences (tokens separated by spaces)",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--unidic-url",
        help="URL to download UniDic-novel ZIP if local MeCab dict lookup fails",
        type=str,
        default=None,
    )

    return vars(parser.parse_args())


def main() -> None:
    args = parse_args()

    if args["verbose"]:
        log.setLevel(logging.DEBUG)
        stdout_handler.setLevel(logging.DEBUG)

    log.addHandler(stdout_handler)

    # if user provided a fallback URL, export for worker subprocesses
    if args.get("unidic_url"):
        url = args["unidic_url"]
        # make the URL & final dir available to children
        os.environ["AOZORA_UNIDIC_URL"] = url
        # download & unpack right now (only once), then export the path
        extract_dir = ensure_unidic_dir(url)
        os.environ["AOZORA_UNIDIC_DIR"] = str(extract_dir)

    # single‐file mode: read one XHTML and dump (a) tokenized, (b) split sentences, or (c) raw text
    if args.get("xhtml_file"):
        # (a) tokenized‐only mode
        if args["tokenized_only"]:
            proc = read_aozora_bunko_xml(
                args["xhtml_file"],
                args["features"],
                args["no_punc"],
                args["speech_mode"],
                args["features_separator"],
                args["features_opening_delim"],
                args["features_closing_delim"],
                do_tokenize=True,
            )
            # one sentence per line, tokens space‐separated; blank line between paragraphs
            for paragraph in proc["paragraphs"]:
                for sentence in paragraph:
                    sys.stdout.write(" ".join(sentence) + "\n")
                sys.stdout.write("\n")
        # (b) split‐sentences plain‐text
        elif args["split_sentences"]:
            proc = read_aozora_bunko_xml(
                args["xhtml_file"],
                args["features"],
                args["no_punc"],
                args["speech_mode"],
                args["features_separator"],
                args["features_opening_delim"],
                args["features_closing_delim"],
                do_tokenize=True,
            )
            out_lines: list[str] = []
            for paragraph in proc["paragraphs"]:
                for sentence in paragraph:
                    out_lines.append("".join(sentence))
                out_lines.append("")
            sys.stdout.write("\n".join(out_lines).rstrip("\n"))
        # (c) raw plain‐text
        else:
            proc = read_aozora_bunko_xml(
                args["xhtml_file"],
                args["features"],
                args["no_punc"],
                args["speech_mode"],
                args["features_separator"],
                args["features_opening_delim"],
                args["features_closing_delim"],
                do_tokenize=False,
            )
            sys.stdout.write(proc["text"])
        sys.exit(0)

    if not args["author_title_csv"] and not args["all"]:
        print('Please specify the "--author-title-csv" or "--all" option.\nAborting.')
        sys.exit(1)

    ndc_tr = make_ndc_map()

    pathlib.Path(args["out"] + "/Tokenized").mkdir(parents=True, exist_ok=True)
    pathlib.Path(args["out"] + "/Plain").mkdir(parents=True, exist_ok=True)

    aozora_db = read_aozora_bunko_list(args["aozora_bunko_repository"], ndc_tr)

    files: list[tuple[str, str, str]] = []
    metadata: list[CorpusMetadata] = []
    if args["all"]:
        for author_ja, titles in aozora_db.items():
            for title, title_dict in titles.items():
                files.append(
                    ("Aozora Bunko", title_dict["file_name"], title_dict["file_path"])
                )
                metadata.append(
                    {
                        "corpus": "Aozora Bunko",
                        "corpus_id": title_dict["file_path"],
                        "author": title_dict["author"],
                        "author_ja": author_ja,
                        "title": title,
                        "title_ja": title_dict.get("title_ja", ""),
                        "brow": "",
                        "genre": "",
                        "narrative_perspective": "",
                        "comments": "",
                        "token_count": None,
                        "author_year": title_dict.get("author_year", ""),
                        "year": title_dict.get("year", ""),
                        "ndc": title_dict.get("ndc", ""),
                    }
                )
    else:
        for csv_path in args["author_title_csv"]:
            fs, db = read_author_title_list(aozora_db, csv_path)
            files.extend(fs)
            metadata.extend(db)

    rejected_files = set()

    token_counts = {}

    if args.get("plaintext_only"):
        for corpus, file_name, file_path in files:
            if corpus != "Aozora Bunko":
                # raw text
                with open(file_path, encoding="utf-8") as f:
                    text = f.read()
            else:
                # reuse HTML-to-text logic (skip tokenization)
                proc = read_aozora_bunko_xml(
                    file_path,
                    args["features"],
                    args["no_punc"],
                    args["speech_mode"],
                    args["features_separator"],
                    args["features_opening_delim"],
                    args["features_closing_delim"],
                    do_tokenize=False,
                )
                text = proc["text"]
            out_path = f"{args['out']}/Plain/{file_name}.txt"
            with open(out_path, "w", encoding="utf-8") as fout:
                fout.write(text)
            log.info(f"Wrote plain text to {out_path}")
        sys.exit(0)

    if args["parallel"]:
        with concurrent.futures.ProcessPoolExecutor() as executor:
            futures = [
                executor.submit(
                    convert_corpus_file,
                    corpus,
                    file_name,
                    file_path,
                    args["out"],
                    args["features"],
                    args["no_punc"],
                    args["speech_mode"],
                    args["min_tokens"],
                    args["features_separator"],
                    args["features_opening_delim"],
                    args["features_closing_delim"],
                )
                for (corpus, file_name, file_path) in files
            ]
            for future in concurrent.futures.as_completed(futures):
                try:
                    result: ConversionResult = future.result()
                    file_name = result["file_name"]
                    file_path = result["file_path"]
                    prefix = result["prefix"]
                    token_count = result["token_count"]
                    rejected = result["rejected"]
                    token_counts[file_path] = token_count
                    if rejected:
                        rejected_files.add(file_path)
                        log.warning(
                            f"{file_name} rejected because it contains < {args['min_tokens']} tokens."
                        )
                    else:
                        log.info(
                            "{} => {}/{{ Tokenized, Plain }}/{}.txt".format(
                                file_path, prefix, file_name
                            )
                        )
                except Exception as e:
                    log.error(
                        "Process {} failed: {}\n\nException: {}".format(
                            future, future.result(), e
                        )
                    )
    else:
        for corpus, file_name, file_path in files:
            result = convert_corpus_file(
                corpus,
                file_name,
                file_path,
                args["out"],
                args["features"],
                args["no_punc"],
                args["speech_mode"],
                args["min_tokens"],
                args["features_separator"],
                args["features_opening_delim"],
                args["features_closing_delim"],
            )
            token_counts[file_path] = result["token_count"]
            if result["rejected"]:
                rejected_files.add(file_path)
                log.warning(
                    f"{file_name} rejected because it contains < {args['min_tokens']} tokens."
                )
            else:
                log.info(
                    "{} => {}/{{ Tokenized, Plain }}/{}.txt".format(
                        file_path, args["out"], file_name
                    )
                )

    rejected_indeces = {
        idx
        for idx, (_, _, file_path) in enumerate(files)
        if file_path in rejected_files
    }

    files = [file for idx, file in enumerate(files) if idx not in rejected_indeces]

    for m in metadata:
        m["token_count"] = token_counts[m["corpus_id"]]

    metadata = [m for idx, m in enumerate(metadata) if idx not in rejected_indeces]

    write_metadata_file(files, metadata, aozora_db, args["out"])


if __name__ == "__main__":
    main()
