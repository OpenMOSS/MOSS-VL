"""CPU regression tests for penalties across preprocessing and sampling."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=8, suite="stage-a-test-cpu")

import ast
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from sglang.srt.sampling.penaltylib.frequency_penalty import BatchedFrequencyPenalizer
from sglang.srt.sampling.penaltylib.orchestrator import BatchedPenalizerOrchestrator
from sglang.srt.sampling.penaltylib.presence_penalty import BatchedPresencePenalizer
from sglang.srt.sampling.penaltylib.repetition_penalty import BatchedRepetitionPenalizer
from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo
from sglang.srt.sampling.sampling_params import TOP_K_ALL


def _load_sampler():
    # Load the actual sampler without importing accelerator/quantization backends.
    path = Path(__file__).resolve().parents[4] / "python/sglang/srt/layers/sampler.py"
    tree = ast.parse(path.read_text())
    cls = next(
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Sampler"
    )
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            cls,
        ],
        type_ignores=[],
    )
    ast.fix_missing_locations(module)
    namespace = {
        "torch": torch,
        "nn": torch.nn,
        "SGLANG_RETURN_ORIGINAL_LOGPROB": False,
    }
    exec(compile(module, str(path), "exec"), namespace)
    return namespace["Sampler"]


Sampler = _load_sampler()


class TestSamplerPenalties(unittest.TestCase):
    def _make_info(self, penalty, greedy):
        reqs = []
        for active in (True, False):
            params = SimpleNamespace(
                frequency_penalty=1.5 if active and penalty == "frequency" else 0.0,
                presence_penalty=1.5 if active and penalty == "presence" else 0.0,
                repetition_penalty=1.5 if active and penalty == "repetition" else 1.0,
            )
            reqs.append(SimpleNamespace(sampling_params=params))
        batch = MagicMock(reqs=reqs, device="cpu")
        orchestrator = BatchedPenalizerOrchestrator(
            3,
            batch,
            {
                BatchedFrequencyPenalizer,
                BatchedPresencePenalizer,
                BatchedRepetitionPenalizer,
            },
        )
        orchestrator.cumulate_output_tokens(torch.tensor([0, 0]))
        orchestrator.cumulate_output_tokens(torch.tensor([2, 2]))
        orchestrator.apply = MagicMock(wraps=orchestrator.apply)
        info = SamplingBatchInfo(
            temperatures=torch.ones(2, 1),
            top_ps=torch.ones(2),
            top_ks=torch.full((2,), TOP_K_ALL, dtype=torch.int32),
            min_ps=torch.zeros(2),
            is_all_greedy=greedy,
            need_top_p_sampling=False,
            need_top_k_sampling=False,
            need_min_p_sampling=False,
            vocab_size=3,
            device="cpu",
            penalizer_orchestrator=orchestrator,
        )
        return info, orchestrator, batch

    def test_penalties_apply_once_in_both_schedules(self):
        for penalty in ("frequency", "presence", "repetition", "none"):
            for overlap in (False, True):
                for greedy in (False, True):
                    with self.subTest(penalty=penalty, overlap=overlap, greedy=greedy):
                        info, orchestrator, _batch = self._make_info(penalty, greedy)
                        if overlap:
                            info = info.copy_for_forward()
                            self.assertIsNone(info.penalizer_orchestrator)

                        logits = torch.tensor([[4.0, 2.0, -4.0], [4.0, 2.0, -4.0]])
                        expected = logits.clone()
                        if penalty in ("frequency", "presence"):
                            expected[0, [0, 2]] -= 1.5
                        elif penalty == "repetition":
                            expected[0, 0] /= 1.5
                            expected[0, 2] *= 1.5

                        # ModelRunner applies penalties before invoking Sampler.
                        info.apply_logits_bias(logits)
                        torch.testing.assert_close(logits, expected)

                        sampler = Sampler.__new__(Sampler)
                        torch.nn.Module.__init__(sampler)
                        sampler.use_nan_detection = False
                        sampler.rl_on_policy_target = None
                        sampler.enable_deterministic = False
                        sampler.use_log_softmax_logprob = False
                        sampler.use_ascend_backend = False
                        sampler._sync_token_ids_across_tp = MagicMock()

                        def sample_probs(probs, *args):
                            torch.testing.assert_close(probs, expected.softmax(dim=-1))
                            return probs.argmax(dim=-1)

                        sampler._sample_from_probs = MagicMock(side_effect=sample_probs)
                        tokens = sampler(
                            SimpleNamespace(next_token_logits=logits),
                            info,
                            False,
                            [],
                            [],
                            torch.tensor([0, 0]),
                        )
                        self.assertEqual(tokens.tolist(), [0, 0])
                        if greedy:
                            torch.testing.assert_close(logits, expected)
                            sampler._sample_from_probs.assert_not_called()
                        else:
                            sampler._sample_from_probs.assert_called_once()

                        expected_calls = int(not overlap and penalty != "none")
                        self.assertEqual(orchestrator.apply.call_count, expected_calls)


if __name__ == "__main__":
    unittest.main()
