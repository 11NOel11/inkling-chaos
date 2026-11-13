"""Write README.md from docs/data.js, so every number in the README is computed, never typed.
Run after make_site_data.py:  python3 code/score/make_readme.py"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
raw = (ROOT / "docs/data.js").read_text()
D = json.loads(raw[raw.index("{"): raw.rindex("}") + 1])
r = lambda x: f"{int(x + 0.5):d}"  # round half up, same as Math.round on the page
A, E, K, N, W = D["arms"], D["effort32k"], D["keep"][0], D["natural"], D["notewins"]
seeds = D.get("seeds")
cmp = {c["name"]: c for c in D["compare"]}
row = lambda n: cmp[n]["arms"] if n in cmp else None

def cell(n, k):
    a = row(n)
    return "—" if not a or a.get(k) is None else f"{r(a[k])}%"

TITLE = "The numbers say chaotic. The database says regular. Inkling 2 Small believes the database."
CITE_TITLE = "The numbers say chaotic, the database says regular: Inkling 2 Small believes the database"
SUBTITLE = "$COST of RL on Tinker fixes the worst of it."
dr = D["dirs"]
lines = [
    '<p align="center">',
    '  <picture>',
    '    <source media="(prefers-color-scheme: dark)" srcset="docs/hero_dark.gif">',
    '    <img alt="A real Lorenz orbit drawn as an ink line over a dot grid; after half a million time units it settles" src="docs/hero_light.gif" width="720">',
    '  </picture>',
    '</p>',
    '',
    f'<h1 align="center">{TITLE}</h1>',
    '',
    f'<p align="center"><b>{SUBTITLE.replace("$COST", "$" + format(D["cost"], ".0f"))}</b></p>',
    '',
    '<p align="center">',
    '  <a href="https://11noel11.github.io/inkling-chaos/"><b>Writeup</b></a> ·',
    '  <a href="#reproduce">Reproduce</a> ·',
    '  <a href="#cite">Cite</a>',
    '</p>',
    '',
    '<p align="center">',
    '  <a href="LICENSE"><img alt="Code: MIT" src="https://img.shields.io/badge/code-MIT-111111"></a>',
    '  <a href="data/LICENSE"><img alt="Data: CC BY 4.0" src="https://img.shields.io/badge/data-CC%20BY%204.0-111111"></a>',
    '</p>',
    '',
    'We show an open model, [Inkling 2 Small](https://huggingface.co/thinkingmachines/Inkling-2-Small), the results of three simulations of a dynamical system, then a one-line claim',
    'that gets them wrong, then a yes/no question. The answer is always in the printed numbers. Because the weights are',
    f'open, we also trained a LoRA adapter on it with RL on Tinker: 40 steps, ${D["cost"]:.2f}.',
    '',
    '## TL;DR',
    '',
    f'- **Thinking harder makes it worse.** Against a claim that misreads the numbers: {r(min(E["arith"]["0.1"]))}–{r(max(E["arith"]["0.1"]))}% right at low reasoning effort, {r(min(E["arith"]["0.99"]))}–{r(max(E["arith"]["0.99"]))}% at maximum.',
    f'- **It listens to who made the claim.** The same wrong claim from "a user": {r(D["dose"]["user"]["base"])}% right. From a "verified reference database": {r(D["dose"]["verified_db"]["base"])}%.',
    '- **$' + f'{D["cost"]:.0f} of RL fixes the worst case.** {r(A["arith"]["base"])}% → {r(A["arith"]["rl"])}% on the misread-numbers claim and {r(A["db"]["base"])}% → {r(A["db"]["rl"])}% on the database claim, over 2,000 questions with unseen wordings; no drop on MMLU.',
    '',
    '**[Read the writeup](https://11noel11.github.io/inkling-chaos/)** for the charts, the model comparison and every caveat.',
    '',
]
lines += [
    '## Reproduce',
    '',
    'With [uv](https://docs.astral.sh/uv/), one command rebuilds every number, the page data and this README from the raw replies:',
    '',
    '```bash',
    'git clone https://github.com/11NOel11/inkling-chaos && cd inkling-chaos',
    'uv run make',
    '```',
    '',
    'Re-running the models: `uv sync --extra eval` (Tinker or Anthropic key, `code/eval/`). Re-training: `uv sync --extra train` (`code/train/evidence_rl.py`). Without uv: `pip install numpy matplotlib && make`.',
    '',
    '| Folder | What is in it |',
    '|---|---|',
    '| `data/` | every question set, every reply with full reasoning, RL metrics |',
    '| `code/` | item generators, eval runners, RL script, scoring |',
    '| `docs/` | the writeup |',
    '',
    'Numbers, predictions and disclosures: [DETAILS.md](DETAILS.md).',
    '',
    '## Credits',
    '',
    'Systems and default parameters from [dysts](https://github.com/williamgilpin/dysts) (William Gilpin, NeurIPS 2021).',
    'Closest related work: [Kumar & Chopra (Lossfunk, NeurIPS 2026)](https://arxiv.org/abs/2609.37616) find the same pull',
    'toward a "verified source" in trivia and reduce it by editing activations; here a short RL run reduces it.',
    'Tinker credit provided by Thinking Machines Lab. Independent work, not affiliated with Thinking Machines.',
    '',
    '## Cite',
    '',
    '```bibtex',
    '@misc{thomas2026inkling,',
    '  author       = {Noel Thomas},',
    f'  title        = {{{CITE_TITLE}}},',
    '  year         = {2026},',
    '  howpublished = {\\url{https://github.com/11NOel11/inkling-chaos}}',
    '}',
    '```',
    '',
]
(ROOT / "README.md").write_text("\n".join(lines))
print("README.md written,", len(lines), "lines")
