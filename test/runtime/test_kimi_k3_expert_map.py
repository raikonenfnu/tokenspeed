"""K3 profile-guided expert-placement validation."""

from __future__ import annotations

import json

import pytest

from tokenspeed.runtime.models.kimi_k3 import (
    _k3_mapped_expert_target,
    _load_k3_expert_map,
)


def test_k3_expert_map_validation(tmp_path) -> None:
    path = tmp_path / "map.json"
    path.write_text(
        json.dumps(
            {
                "logical_to_physical": [
                    [0, 1, 2, 3],
                    [2, 0, 3, 1],
                ]
            }
        )
    )
    rows = _load_k3_expert_map(str(path), 2, 4, 2)
    assert rows == ((0, 1, 2, 3), (2, 0, 3, 1))

    path.write_text(
        json.dumps(
            {
                "logical_to_physical": [
                    [0, 1, 2, 3],
                    [0, 0, 2, 3],
                ]
            }
        )
    )
    _load_k3_expert_map.cache_clear()
    with pytest.raises(ValueError, match="permutation"):
        _load_k3_expert_map(str(path), 2, 4, 2)


@pytest.mark.parametrize(
    ("name", "ep_rank", "expected"),
    [
        (
            "model.layers.1.block_sparse_moe.experts.0.w1.weight",
            1,
            (
                "model.layers.1.block_sparse_moe.experts.w13_weight",
                "w1",
                0,
            ),
        ),
        (
            "model.layers.1.block_sparse_moe.experts.3.w3.weight_scale",
            0,
            (
                "model.layers.1.block_sparse_moe.experts.w13_weight_scale",
                "w3",
                1,
            ),
        ),
        (
            "model.layers.1.block_sparse_moe.experts.1.w2.weight",
            0,
            None,
        ),
    ],
)
def test_k3_mapped_expert_target(name, ep_rank, expected) -> None:
    # Logical [0,1,2,3] -> physical [2,3,0,1].
    rows = ((0, 1, 2, 3), (2, 3, 0, 1))
    handled, target = _k3_mapped_expert_target(
        name,
        rows,
        ep_rank=ep_rank,
        ep_size=2,
    )
    assert handled
    assert target == expected


def test_k3_mapped_expert_target_ignores_shared_expert() -> None:
    handled, target = _k3_mapped_expert_target(
        "model.layers.1.block_sparse_moe.shared_experts.down_proj.weight",
        ((0, 1), (0, 1)),
        ep_rank=0,
        ep_size=1,
    )
    assert not handled
    assert target is None
