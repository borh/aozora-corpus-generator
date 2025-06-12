import sys
import zipfile
from pathlib import Path

import pytest

from aozora_corpus_generator.cli import main

DATA_DIR = Path("tests/data")
CORPUS_DIR = DATA_DIR / "test_corpus"
EXPECTED_DIR = DATA_DIR / "test_results"


def test_end2end_compare_results(tmp_path):
    """
    Runs the full CLI pipeline on our test corpus and compares
    both Plain and Tokenized outputs against the precomputed results.
    """
    # 1) build a dummy Aozora‐index ZIP with only the CSV header
    repo_zip = tmp_path / "dummy_repo.zip"
    with zipfile.ZipFile(repo_zip, "w") as zf:
        zf.writestr(
            "list_person_all_extended_utf8.csv",
            "作品名,作品名読み,副題,副題読み,姓,名,姓ローマ字,名ローマ字,生年月日,没年月日,底本初版発行年1,初出,文字遣い種別,XHTML/HTMLファイルURL,分類番号\n",
        )

    out_dir = tmp_path / "out"
    sys.argv = [
        "prog",
        "--author-title-csv",
        str(CORPUS_DIR / "test.csv"),
        "--aozora-bunko-repository",
        str(repo_zip),
        "--min-tokens",
        "10",
        "--out",
        str(out_dir),
    ]

    # run the full‐corpus pipeline
    main()

    # now compare every generated file against the golden copy
    for phase in ("Plain", "Tokenized"):
        exp_dir = EXPECTED_DIR / phase
        act_dir = out_dir / phase
        for exp in exp_dir.glob("*.txt"):
            act = act_dir / exp.name
            assert act.exists(), f"Missing {phase} output: {act}"
            expected_text = exp.read_text(encoding="utf-8").strip()
            actual_text = act.read_text(encoding="utf-8").strip()
            assert actual_text == expected_text, f"Mismatch in {phase}/{exp.name}"


def test_single_xhtml_file_stdout(capsys):
    """
    For each HTML in test_corpus, run in single xhtml file mode and compare stdout
    against the golden Plain/<stem>.txt in test_results.
    """
    for html_path in sorted(CORPUS_DIR.glob("*.html")):
        sys.argv = ["prog", "--xhtml-file", str(html_path), "--split-sentences"]
        with pytest.raises(SystemExit) as e:
            main()
        assert e.value.code == 0
        actual = capsys.readouterr().out.strip()
        gold = EXPECTED_DIR / "Plain" / f"{html_path.stem}.txt"
        assert gold.exists(), f"Missing expected {gold}"
        expected = gold.read_text(encoding="utf-8").strip()
        assert actual == expected, f"Mismatch for {html_path.name}"
