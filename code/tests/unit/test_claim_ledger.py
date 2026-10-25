"""A reflection's sentences are registered as claims and settled by later readings."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from rfs.contracts.evolution import claim_settles, normalize_claims, normalize_reflection
from rfs.evolution.direct.claims import CLAIM_SETTLEMENT_LIMIT, ClaimLedger, measured_metrics


def _claim(text: str, metric: str = "interface:box_pass", op: str = "<=", value: float = 2) -> dict:
    return {"claim": text, "apply_at": "ppt_compilation", "criterion": {"metric": metric, "op": op, "value": value}}


class ClaimContractTests(unittest.TestCase):
    def test_a_claim_is_settled_only_by_the_reading_it_named(self):
        claim = normalize_claims([_claim("the crossings fall")])[0]

        self.assertEqual(claim_settles(claim, {"interface:box_pass": 1}), "confirmed")
        self.assertEqual(claim_settles(claim, {"interface:box_pass": 4}), "refuted")
        # A round that did not measure the number settles nothing, so no verdict is recorded.
        self.assertEqual(claim_settles(claim, {"overall_score": 1.0}), "")
        self.assertEqual(claim_settles(claim, None), "")

    def test_a_claim_with_no_reading_is_kept_as_a_note_and_settles_nothing(self):
        claim = normalize_claims([{"claim": "the seat was confused"}])[0]

        self.assertEqual(claim["criterion"], {})
        self.assertEqual(claim_settles(claim, {"overall_score": 1.0}), "")

    def test_a_bad_operator_or_value_does_not_make_a_reading(self):
        claim = normalize_claims([{"claim": "x", "criterion": {"metric": "m", "op": "~=", "value": 1}}])[0]

        self.assertEqual(claim["criterion"], {})

    def test_the_same_sentence_against_the_same_reading_is_one_claim(self):
        claims = normalize_claims([_claim("the crossings fall"), _claim("the crossings fall")])

        self.assertEqual(len(claims), 1)

    def test_the_reflection_contract_carries_the_claims(self):
        reflection = normalize_reflection({"outcome": "improved", "claims": [_claim("the crossings fall")]})

        self.assertEqual(len(reflection["claims"]), 1)
        self.assertEqual(reflection["claims"][0]["apply_at"], "ppt_compilation")


class MeasuredMetricsTests(unittest.TestCase):
    def test_only_measured_numbers_are_offered_to_a_claim(self):
        result = {
            "stage_feedback": [
                {"severity": "blocker", "defect_key": "interface:box_pass"},
                {"severity": "blocker", "defect_key": "interface:box_pass"},
                {"severity": "hint", "defect_key": "not_a_blocker"},
            ],
            "dimension_scores": {"readability": 0.8, "science": "n/a"},
            "overall_score": 0.9,
        }

        metrics = measured_metrics(result, extra={"round_score": 1})

        self.assertEqual(metrics["interface:box_pass"], 2.0)
        self.assertNotIn("not_a_blocker", metrics)
        self.assertEqual(metrics["dimension:readability"], 0.8)
        self.assertNotIn("dimension:science", metrics)
        self.assertEqual(metrics["overall_score"], 0.9)
        self.assertEqual(metrics["round_score"], 1.0)


class ClaimLedgerTests(unittest.TestCase):
    def _ledger(self, root: Path) -> ClaimLedger:
        return ClaimLedger(root / "library")

    def test_a_claim_is_settled_by_a_later_reading_and_the_words_survive(self):
        with tempfile.TemporaryDirectory() as name:
            ledger = self._ledger(Path(name))
            ledger.register(normalize_claims([_claim("the crossings fall")]), round_index=3)

            settled = ledger.settle_open({"interface:box_pass": 1}, round_index=4, source_run="paper-a")

            self.assertEqual([item["event"] for item in settled], ["claim_confirmed"])
            report = ledger.report()
            self.assertEqual(report["counts"]["confirmed"], 1)
            self.assertEqual(report["claims"][0]["claim"], "the crossings fall")
            self.assertEqual(report["counts"]["open"], 0)

    def test_a_reading_that_was_not_taken_settles_nothing(self):
        with tempfile.TemporaryDirectory() as name:
            ledger = self._ledger(Path(name))
            ledger.register(normalize_claims([_claim("the crossings fall")]), round_index=3)

            settled = ledger.settle_open({"overall_score": 0.5}, round_index=4)

            self.assertEqual(settled, [])
            self.assertEqual(ledger.report()["counts"]["open"], 1)
            self.assertEqual(ledger.refutation_streak(ledger.open_claims()[0]["claim_id"]), 0)

    def test_noise_is_not_a_verdict(self):
        with tempfile.TemporaryDirectory() as name:
            ledger = self._ledger(Path(name))
            ledger.register(normalize_claims([_claim("the crossings fall")]), round_index=3)
            identifier = ledger.open_claims()[0]["claim_id"]

            self.assertEqual(ledger.settle(identifier, "noise", round_index=4), {})
            self.assertEqual(ledger.refutation_streak(identifier), 0)

    def test_consecutive_refutations_write_a_claim_off_and_a_confirmation_clears_it(self):
        with tempfile.TemporaryDirectory() as name:
            ledger = self._ledger(Path(name))
            ledger.register(normalize_claims([_claim("the crossings fall")]), round_index=1)
            identifier = ledger.open_claims()[0]["claim_id"]

            ledger.settle(identifier, "refuted", round_index=2)
            self.assertEqual(ledger.disabled_ids(), set())
            ledger.settle(identifier, "confirmed", round_index=3)
            self.assertEqual(ledger.refutation_streak(identifier), 0)
            ledger.settle(identifier, "refuted", round_index=4)
            ledger.settle(identifier, "refuted", round_index=5)

            self.assertEqual(ledger.refutation_streak(identifier), CLAIM_SETTLEMENT_LIMIT)
            self.assertIn(identifier, ledger.disabled_ids())
            self.assertTrue(ledger.report()["claims"][0]["disabled"])

    def test_a_claim_is_not_registered_twice(self):
        with tempfile.TemporaryDirectory() as name:
            ledger = self._ledger(Path(name))
            claims = normalize_claims([_claim("the crossings fall")])

            self.assertEqual(len(ledger.register(claims, round_index=1)), 1)
            self.assertEqual(ledger.register(claims, round_index=2), [])
            self.assertEqual(ledger.report()["counts"]["registered"], 1)


if __name__ == "__main__":
    unittest.main()
