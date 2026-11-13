"""LoRA RL on Inkling 2 Small: read the diagnostics in the prompt instead of deferring to a wrong note.

Items: data/items/inkling_rl/{train,test_id,test_ood}.jsonl (code/gen/inkling_rl_items.py).
Reward: 1 if the last TRUE/FALSE in the reply matches, else 0; -0.1 format penalty if no TRUE/FALSE / truncated.
Test metrics are logged per tag "<split>-<kind>-<rel>" so contradict / agree / none are separate.

Usage: uv run python evidence_rl.py log_path=runs/evid_smoke max_steps=2 [learning_rate=4e-5 eval_every=10 ...]
"""
import asyncio
import json
import random
import re
from functools import partial
from pathlib import Path

import chz

from tinker_cookbook import renderers
from tinker_cookbook.rl import train
from tinker_cookbook.rl.problem_env import ProblemEnv, ProblemGroupBuilder
from tinker_cookbook.rl.types import RLDataset, RLDatasetBuilder
from tinker_cookbook.tokenizer_utils import get_tokenizer

MODEL = "thinkingmachines/Inkling-2-Small"
ITEMS = Path(__file__).resolve().parents[2] / "data" / "items" / "inkling_rl"
EFFORT = 0.99


def parse(text):
    hits = re.findall(r"\b(TRUE|FALSE)\b", text or "") or re.findall(r"\b(true|false)\b", text or "", re.I)
    return None if not hits else hits[-1].upper() == "TRUE"


class EvidenceEnv(ProblemEnv):
    def __init__(self, renderer, prompt, answer, effort=True):
        super().__init__(renderer, format_coef=0.1)
        self.prompt, self.answer, self.effort = prompt, answer, effort

    async def initial_observation(self):
        convo = [{"role": "user", "content": self.prompt}]
        if self.effort:  # tml renderers take a reasoning-effort argument; others use their default thinking mode
            return self.renderer.build_generation_prompt(convo, effort=EFFORT), self.stop_condition
        return self.renderer.build_generation_prompt(convo), self.stop_condition

    def get_question(self):
        return self.prompt

    def check_answer(self, sample_str):
        return parse(sample_str) == self.answer

    def check_format(self, sample_str):
        return parse(sample_str) is not None

    def get_reference_answer(self):
        return "TRUE" if self.answer else "FALSE"


class EvidenceDataset(RLDataset):
    def __init__(self, rows, batch_size, group_size, renderer, tag_by_cell, effort=True):
        self.rows, self.batch_size, self.group_size = rows, batch_size, group_size
        self.renderer, self.tag_by_cell, self.effort = renderer, tag_by_cell, effort

    def get_batch(self, index):
        out = []
        for r in self.rows[index * self.batch_size:(index + 1) * self.batch_size]:
            tag = f"{r['id'].split('-')[0]}-{r['kind']}-{r['rel'] or 'na'}" if self.tag_by_cell else "train"
            out.append(ProblemGroupBuilder(env_thunk=partial(EvidenceEnv, self.renderer, r["prompt"], r["answer"], self.effort),
                                           num_envs=self.group_size, dataset_name=tag))
        return out

    def __len__(self):
        return (len(self.rows) + self.batch_size - 1) // self.batch_size


@chz.chz
class EvidenceBuilder(RLDatasetBuilder):
    batch_size: int = 32
    group_size: int = 4
    n_test: int = 200  # per test split
    seed: int = 0
    model: str = MODEL  # comparison runs pass another Tinker model

    async def __call__(self):
        from tinker_cookbook import model_info
        rname = model_info.get_recommended_renderer_name(self.model)
        renderer = renderers.get_renderer(rname, tokenizer=get_tokenizer(self.model), model_name=self.model)
        eff = rname.startswith("tml")
        load = lambda s: [json.loads(l) for l in open(ITEMS / f"{s}.jsonl")]  # noqa: E731
        tr = load("train")
        random.Random(self.seed).shuffle(tr)
        te = load("test_id")[: self.n_test] + load("test_ood")[: self.n_test]
        return (EvidenceDataset(tr, self.batch_size, self.group_size, renderer, False, eff),
                EvidenceDataset(te, len(te), 1, renderer, True, eff))



def _sampler_only_final():
    """Save only sampler weights for the final checkpoint (free storage: the 46.6 GB training state is not needed to
    sample or evaluate the adapter; Noel approved dropping it, 2026-11-13)."""
    from tinker_cookbook import checkpoint_utils as cu
    for fname in ("save_checkpoint", "save_checkpoint_async"):
        orig = getattr(cu, fname)

        def wrap(*args, _orig=orig, **kw):
            if kw.get("name") == "final" and kw.get("kind") == "both":
                kw["kind"] = "sampler"
            return _orig(*args, **kw)
        setattr(cu, fname, wrap)

def main():
    bp = chz.Blueprint(train.Config).apply({
        "model_name": MODEL, "recipe_name": "evidence_rl", "renderer_name": "tml_v0",
        "log_path": "runs/evidence_rl", "dataset_builder": EvidenceBuilder, "learning_rate": 4e-5,
        "max_tokens": 12000, "eval_every": 10, "save_every": 10, "lora_rank": 32, "temperature": 1.0,
    })
    import sys
    import os
    if os.environ.get("FULL_FINAL") != "1":  # FULL_FINAL=1 keeps the training state (needed for push-hf)
        _sampler_only_final()
    cfg = bp.make_from_argv(sys.argv[1:])
    asyncio.run(train.main(cfg))


if __name__ == "__main__":
    main()
