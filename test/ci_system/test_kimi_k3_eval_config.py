import json
import shlex
from pathlib import Path

import yaml

CONFIG_PATH = (
    Path(__file__).resolve().parents[1]
    / "ci"
    / "eval"
    / "kimi-k3-mxfp4-tp8ep8-evalscope-aime26-amd.yaml"
)


def _option_values(argv: list[str], option: str, count: int = 1) -> list[str]:
    start = argv.index(option) + 1
    return argv[start : start + count]


def test_kimi_k3_eval_is_seeded_and_decode_graph_only():
    config = yaml.safe_load(CONFIG_PATH.read_text())
    server_args = shlex.split(config["server"]["command"])
    eval_args = shlex.split(config["eval"]["command"])

    assert config["env"]["TOKENSPEED_FLAT_KV"] == "ON"
    assert _option_values(server_args, "--sampling-backend") == ["triton"]
    assert _option_values(server_args, "--seed") == ["42"]
    assert "--force-deterministic-rsag" not in server_args
    assert "--no-enable-prefix-caching" not in server_args

    assert "--disable-prefill-graph" in server_args
    assert _option_values(server_args, "--cudagraph-capture-sizes", 5) == [
        "1",
        "2",
        "4",
        "8",
        "16",
    ]

    generation_config = json.loads(_option_values(eval_args, "--generation-config")[0])
    assert generation_config["do_sample"] is True
    assert generation_config["temperature"] == 1.0
    assert generation_config["top_p"] == 0.95
    assert generation_config["seed"] == 42
    assert generation_config["max_tokens"] == 32768


def test_kimi_k3_eval_flushes_prefix_cache_before_each_attempt():
    config = yaml.safe_load(CONFIG_PATH.read_text())
    eval_command = config["eval"]["command"]

    assert "http://127.0.0.1:8001/flush_cache" in eval_command
    assert eval_command.index("/flush_cache") < eval_command.index("evalscope eval")


def test_kimi_k3_evalscope_version_is_pinned():
    config = yaml.safe_load(CONFIG_PATH.read_text())
    assert any(
        "evalscope[perf]==1.9.1" in command for command in config["eval"]["install"]
    )
