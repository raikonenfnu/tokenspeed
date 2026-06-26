# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Regression tests for issue #2: TP=8 Iris fd-passing deadlock.

The story
---------
When ``dist.init_process_group`` is called *without* ``device_id``, NCCL falls
back to **guessing each rank's GPU from its global rank** (global rank ``r`` ->
GPU ``r``). That guess is correct only when the rank->GPU map is contiguous. It
is not, the moment a job spans more than one node: tokenspeed assigns each
rank's GPU as

    gpu_id = global_rank % nprocs_per_node + base_gpu_id          (event_loop.py)

so on a 2-node x 4-GPU TP=8 job, global ranks 4..7 run on local GPUs 0..3 --
their global rank no longer equals the GPU they own. NCCL then guesses the wrong
device, and the first object collective on the default group (``all_gather_object``
inside Iris symmetric-heap fd-passing) deadlocks. (TP=4 on a single node is
contiguous, which is why it never hit this.)

The fix stops guessing: every rank already knows the GPU it owns (``gpu_id``),
so we hand that device to ``init_process_group`` explicitly.

These tests reproduce that exact map with tokenspeed's own formula and assert we
bind the GPU each rank *owns*, not the GPU NCCL would *guess* from the global
rank. They need no GPUs, fail on the pre-fix code, and pass with the fix.
"""

from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from tokenspeed.runtime.distributed.process_group_manager import ProcessGroupManager

# A realistic 2-node x 4-GPU TP=8 layout, using tokenspeed's assignment
# ``gpu_id = global_rank % nprocs_per_node`` (base_gpu_id = 0). Ranks 0..3 (node
# 0) own GPUs 0..3; ranks 4..7 (node 1) own GPUs 0..3 again -- so for the second
# node ``global_rank != gpu_id`` and NCCL's global-rank guess is wrong.
_NPROCS_PER_NODE = 4
_WORLD_SIZE = 8
_RANK_TO_OWNED_GPU = {r: r % _NPROCS_PER_NODE for r in range(_WORLD_SIZE)}


def _nccl_would_guess(global_rank: int) -> torch.device:
    """The GPU NCCL infers with no ``device_id`` -- the global rank itself.
    On the second node this is the *wrong* GPU."""
    return torch.device("cuda", global_rank)


@pytest.mark.parametrize("global_rank, owned_gpu", sorted(_RANK_TO_OWNED_GPU.items()))
def test_process_group_binds_owned_gpu(global_rank, owned_gpu):
    """``init_distributed`` must hand ``init_process_group`` the GPU this rank
    owns, never letting NCCL fall back to the global-rank guess."""
    owned = torch.device("cuda", owned_gpu)

    with mock.patch(
        "tokenspeed.runtime.distributed.process_group_manager.dist"
    ) as mdist:
        mdist.is_initialized.return_value = False
        ProcessGroupManager().init_distributed(
            SimpleNamespace(world_size=_WORLD_SIZE, rank=global_rank),
            distributed_init_method="tcp://127.0.0.1:12345",
            backend="nccl",
            timeout=1800,
            device_id=owned,
        )

    _, kwargs = mdist.init_process_group.call_args
    bound = kwargs.get("device_id")
    assert bound is not None, (
        "init_process_group got no device_id -> NCCL guesses the GPU from the "
        "global rank and deadlocks all_gather_object at TP=8."
    )
    assert bound == owned, "must bind the GPU this rank owns"
    # Second node (ranks 4..7): owned GPU != global rank, so binding the owned
    # device is precisely what avoids NCCL's wrong guess.
    if owned != _nccl_would_guess(global_rank):
        assert bound != _nccl_would_guess(global_rank)


def test_initializer_binds_assigned_gpu_for_this_rank():
    """End-to-end through the real caller: ``DistributedInitializer.initialize``
    derives ``device_id`` from the rank's *assigned* GPU (``gpu_id``), not its
    global rank. Modelled on global rank 6 of a 2x4 TP=8 job, which owns local
    GPU 2 -- so a regression to the rank-based guess (GPU 6) is caught."""
    from tokenspeed.runtime.execution import distributed_initializer as di

    global_rank = 6
    owned_gpu = global_rank % _NPROCS_PER_NODE  # == 2
    config = di.DistributedConfig(
        device="cuda",
        gpu_id=owned_gpu,
        world_size=_WORLD_SIZE,
        global_rank=global_rank,
        local_rank=owned_gpu,
        attn_tp_rank=0,
        attn_tp_size=_WORLD_SIZE,
        dp_size=1,
        dense_tp_size=_WORLD_SIZE,
        moe_ep_size=1,
        moe_ep_rank=0,
        nccl_port=12345,
        nnodes=2,
        nprocs_per_node=_NPROCS_PER_NODE,
        mapping=mock.MagicMock(),
    )

    with mock.patch.object(di, "pg_manager") as mpg, mock.patch.object(
        di.torch, "get_device_module"
    ) as mdev, mock.patch.object(
        di, "maybe_set_numa_aware_cpu_affinity"
    ), mock.patch.object(
        di, "get_available_gpu_memory", return_value=10.0
    ):
        mdev.return_value = mock.MagicMock()  # set_device(...) becomes a no-op
        di.DistributedInitializer.initialize(config)

    _, kwargs = mpg.init_distributed.call_args
    assert kwargs.get("device_id") == torch.device("cuda", owned_gpu), (
        "initialize must bind this rank's assigned GPU (gpu_id=2), not its "
        "global rank (6); otherwise TP=8 deadlocks in Iris fd-passing."
    )
    assert kwargs.get("device_id") != _nccl_would_guess(global_rank)
