from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

_REPO_ROOT = Path(__file__).parents[3]
_CLIENT_PATH = _REPO_ROOT / "examples" / "moss_vl_realtime_client.py"
_SPEC = importlib.util.spec_from_file_location("moss_vl_realtime_client", _CLIENT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_CLIENT = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_CLIENT)


def test_manifest_inputs_preserve_prompts_events_and_one_fps(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.jsonl"
    case = {
        "case_id": "case-1",
        "system_prompt": "system",
        "initial_prompt": "watch",
        "frame_interval_seconds": 1.0,
        "events": [
            {
                "type": "frame",
                "seq_no": 0,
                "timestamp": 0.0,
                "frame_path": "/frame.png",
                "final": False,
            },
            {
                "type": "prompt",
                "seq_no": 1,
                "prompt": "How many?",
                "final": True,
            },
        ],
    }
    manifest.write_text(json.dumps(case) + "\n")
    args = SimpleNamespace(
        manifest=manifest,
        case_id="case-1",
        frame=None,
        timestamp=None,
        frame_interval=None,
        fps=None,
        prompt="unused",
    )

    config, events, interval = _CLIENT.resolve_inputs(args)

    assert config == {"prompt": "watch", "system_prompt": "system"}
    assert events == case["events"]
    assert interval == 1.0
