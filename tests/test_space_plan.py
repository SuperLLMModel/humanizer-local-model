# -*- coding: utf-8 -*-
"""Tests for the Space's GPU-slot sizing (space/app.py, plan()) and its cut-off notices.

    HUMANIZER_MOCK=1 python3 -m unittest tests.test_space_plan -v

Needs gradio (space/requirements.txt); app.py is imported in mock mode, without torch or the model.
"""
import os
import sys
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPACE = os.path.join(ROOT, "space")
sys.path.insert(0, SPACE)
os.environ["HUMANIZER_MOCK"] = "1"
os.environ["HUMANIZER_MOCK_FAULTS"] = "cut"     # each part's first sample stops halfway (as at max_time)
os.environ.pop("HF_TOKEN", None)                # never write to the usage dataset from a test

try:
    import gradio  # noqa: F401
    _cwd = os.getcwd()
    os.chdir(SPACE)
    import app  # noqa: E402
    os.chdir(_cwd)
    HAVE_APP = True
except ImportError:
    HAVE_APP = False

PLAIN_TPS_P5 = 15.8      # plain transformers generate on ZeroGPU, usage log 10-02..10-10, p5 of 374 outputs
OUT_RATIO_P99 = 1.39     # output / draft tokens, drafts of 50+ tokens


def old_plan(n_tok):
    """plan() before 2026-10-19."""
    max_new = max(256, min(2048, round(2.5 * n_tok)))
    budget = min(108, max(18, 6 + (1.5 * n_tok + 64) / 16.0))
    return max_new, round(budget, 1), int(budget + 12)


@unittest.skipUnless(HAVE_APP, "gradio not installed")
class PlanTest(unittest.TestCase):
    RANGE = range(1, app.MAX_DRAFT_TOKENS + 1) if HAVE_APP else ()

    def test_max_new_is_the_apps_rule(self):
        for n in (1, 50, 102, 103, 500, 819, 820, 1400):
            self.assertEqual(app.plan(n)[0], max(256, min(2048, round(2.5 * n))), n)

    def test_slot_bounds_and_overhead(self):
        for n in self.RANGE:
            max_new, budget, slot = app.plan(n)
            self.assertGreaterEqual(slot, app.SLOT_MIN, n)
            self.assertLessEqual(slot, app.SLOT_MAX, n)
            self.assertGreaterEqual(slot, budget + app.SLOT_OVERHEAD - 1e-9, n)
            self.assertLess(slot, budget + app.SLOT_OVERHEAD + 1, n)
            self.assertGreater(budget, 0, n)

    def test_monotone(self):
        prev = (0, 0, 0)
        for n in self.RANGE:
            cur = app.plan(n)
            self.assertTrue(all(c >= p for c, p in zip(cur, prev)), (n, prev, cur))
            prev = cur

    def test_slow_path_finishes_a_p99_rewrite(self):
        """A worker without the AOTI step (plain generate, at its p5 speed) must not be cut off before a p99-length
        rewrite is done, except where the slot is at ZeroGPU's ceiling."""
        for n in self.RANGE:
            _, budget, _ = app.plan(n)
            if budget >= app.SLOT_MAX - app.SLOT_OVERHEAD:
                continue
            need = 0.3 + (OUT_RATIO_P99 * n) / PLAIN_TPS_P5
            self.assertGreaterEqual(budget, need, n)

    def test_floor_covers_max_new_floor_on_slow_path(self):
        _, budget, _ = app.plan(1)
        self.assertGreaterEqual(budget, 256 / 16.0)

    def test_never_asks_for_more_than_before(self):
        for n in self.RANGE:
            self.assertLessEqual(app.plan(n)[2], old_plan(n)[2], n)

    def test_known_values(self):
        # median draft (216 tokens), 500 tokens, p90 (925 tokens), a full 700-word part, the token backstop
        self.assertEqual(app.plan(216), (540, 28.6, 35))
        got = {n: app.plan(n)[2] for n in (10, 50, 100, 500, 925, 1100, 1400)}
        self.assertEqual(got, {10: 22, 50: 22, 100: 25, 500: 61, 925: 99, 1100: 115, 1400: 120})


@unittest.skipUnless(HAVE_APP, "gradio not installed")
class CutNoticeTest(unittest.TestCase):
    """A sample that stops at max_time is never shown silently: the panel says which part is cut off."""

    def run_rewrite(self, draft):
        app._mock_seen.clear()
        last = None
        with mock.patch("time.sleep"):
            for last in app.rewrite(draft, "[]"):
                pass
        return last

    def test_single_part(self):
        draft = ("The quarterly report shows revenue grew 12% to $4.2 million. Costs rose 3%. "
                 "The team hired 5 engineers in March. Next quarter the focus is retention. ") * 3
        last = self.run_rewrite(draft)
        self.assertIn('data-state="done"', last)
        self.assertIn("time limit, so the ending is cut off", last)

    def test_long_draft_names_the_part(self):
        para = ("The field team logged 41 site visits across the region during the spring survey. "
                "Each visit recorded soil moisture, canopy cover and the count of seedlings per plot. ")
        draft = "\n\n".join(para * 12 for _ in range(3))      # about 1,000 words: two parts
        self.assertGreater(len(app._parts(draft)), 1)
        last = self.run_rewrite(draft)
        self.assertIn('data-state="done"', last)
        self.assertRegex(last, r"Parts? [0-9, and]+ of \d+ stopped at this demo")
        self.assertIn("段（共", last)


if __name__ == "__main__":
    unittest.main()
