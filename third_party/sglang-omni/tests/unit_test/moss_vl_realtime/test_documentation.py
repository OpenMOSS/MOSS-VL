"""Keep bilingual entry points and published reference numbers consistent."""

import re
from pathlib import Path

import pytest
from markdown_it import MarkdownIt

ROOT = Path(__file__).resolve().parents[3]
README_DIRS = (
    ".",
    "benchmarks",
    "benchmarks/tts_serving",
    "deployment/moss_vl_realtime",
    "deployment/repro",
    "docs",
    "examples",
    "playground",
    "playground/qwen-omni/realtime",
    "sglang_omni/models/audar_tts",
    "sglang_omni/models/fishaudio_s2_pro",
    "sglang_omni/models/minimax_music3",
    "tests",
)


@pytest.mark.parametrize("directory", README_DIRS)
def test_readme_language_pair(directory):
    english = (ROOT / directory / "README.md").read_text()
    chinese = (ROOT / directory / "README_zh.md").read_text()
    assert "[简体中文](./README_zh.md)" in english
    assert "[English](./README.md)" in chinese
    assert len(re.findall(r"[\u4e00-\u9fff]", chinese)) > 100


def table_numbers(document):
    numbers = []
    in_body = False
    for token in MarkdownIt().enable("table").parse(document):
        if token.type == "tbody_open":
            in_body = True
        elif token.type == "tbody_close":
            in_body = False
        elif in_body and token.type == "inline":
            numbers.extend(re.findall(r"\d+(?:\.\d+)?", token.content))
    return numbers


def test_reference_table_numbers_match_between_languages():
    directory = ROOT / "deployment/moss_vl_realtime"
    english = table_numbers((directory / "README.md").read_text())
    chinese = table_numbers((directory / "README_zh.md").read_text())
    assert english
    assert english == chinese


def test_installation_uses_the_same_lock_and_toolkit_in_both_languages():
    for name in ("installation.md", "installation_zh.md"):
        text = (ROOT / "docs/get_started" / name).read_text()
        for path in (
            "deployment/repro/requirements.lock",
            "deployment/repro/build-constraints.txt",
            "deployment/repro/cuda_toolkit.py",
            "deployment/moss_vl_realtime/check_env.py",
            "deployment/moss_vl_realtime/start.sh",
        ):
            assert path in text
            assert (ROOT / path).is_file()
        assert "--require-hashes" in text
        assert "OpenMOSS-Team/MOSS-VL-Realtime-SGLANG" in text
