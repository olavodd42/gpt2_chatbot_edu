import json
import sys
from pathlib import Path
import pytest

# Ensure the package can be imported when running tests directly
sys.path.append(str(Path(__file__).resolve().parents[1]))
from src import templates


def test_json_to_text(tmp_path, monkeypatch):
    sample = {
        "dialogue": [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
        ]
    }
    sample_file = tmp_path / "sample.jsonl"
    sample_file.write_text(json.dumps(sample) + "\n")
    monkeypatch.setattr(templates.os.path, "join", lambda base, name: sample_file)
    result = templates.json_to_text("sample.jsonl")
    expected = (
        f"\n{templates.USER}: Hello\n"
        f"{templates.ASSISTANT}: Hi\n"
        f"{templates.ASSISTANT}: "
    )
    assert result == expected
