"""Test the tests: put one deliberate bug at a time into a copy of the code, and check a test fails.

    uv run python scripts/mutate.py          # 29 mutants; a few minutes

Each mutant is a small, plausible mistake in a key code path. A mutant "survives" if the whole test
suite still passes with it — then a behaviour that matters is untested. The repository itself is never
modified: the work happens in a temporary copy, which `PYTHONPATH` puts ahead of the installed package.
A mutant that makes the tests hang (e.g. an exponent cap removed, so 9**9**9 never finishes) counts as
caught after a timeout.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

MUTANTS = [  # (file, original, mutant)
    ("byoai/tokenizer.py", "key=lambda p: (pair_count[p], (-p[0], -p[1]))", "key=lambda p: (pair_count[p], (p[0], p[1]))"),
    ("byoai/tokenizer.py", "sorted(self.special, key=len, reverse=True)", "sorted(self.special, key=len)"),
    ("byoai/tokenizer.py", "if len(self._cache) < 100_000:", "if len(self._cache) < 0:"),
    ("byoai/autograd.py", "grad = grad.sum(axis=axis, keepdims=True)", "grad = grad"),
    ("byoai/autograd.py", "np.add.at(", "np.put("),
    ("byoai/autograd.py", "self.grad = g if self.grad is None else self.grad + g", "self.grad = g"),
    ("byoai/autograd.py", "if any(node._used for node in order):", "if False:"),
    ("byoai/gpt.py", "* (1.0 / math.sqrt(D))", "* 1.0"),
    ("byoai/gpt.py", "scores = scores.masked_fill(self._mask[:T, :T], -1e9)", "scores = scores"),
    ("byoai/gpt.py", "if norm > max_norm:", "if norm > 1e9:"),
    ("byoai/gpt.py", "proj_std = 0.02 / math.sqrt(2 * L)", "proj_std = 0.02"),
    ("byoai/generate.py", "future = np.arange(start + T)[None, :] > (start + np.arange(T))[:, None]", "future = np.arange(start + T)[None, :] > (np.arange(T))[:, None]"),
    ("byoai/generate.py", "context = (context + [nxt])[-window // 2 :] if full else context + [nxt]", "context = (context + [nxt])[-window // 4 :] if full else context + [nxt]"),
    ("byoai/generate.py", "cut = int(np.searchsorted(cumulative, top_p)) + 1", "cut = int(np.searchsorted(cumulative, top_p))"),
    ("byoai/vectordb.py", "if any(row[i] < dists[j] for i in chosen):", "if False:"),
    ("byoai/vectordb.py", "if dist_c > -best[0][0] and len(best) >= ef:", "if len(best) >= ef:"),
    ("byoai/vectordb.py", "live = [(d, i) for d, i in with_copies if i not in self.deleted][:k]", "live = with_copies[:k]"),
    ("byoai/vectordb.py", "if first != node:", "if False:"),
    ("byoai/rag.py", "s[ids] += self.idf[w] * tf * (self.k1 + 1) / (tf + norm[ids])", "s[ids] += tf * (self.k1 + 1) / (tf + norm[ids])"),
    ("byoai/rag.py", "start = max(end - overlap, start + 1)", "start = end"),
    ("byoai/rag.py", "fused[i] = fused.get(i, 0.0) + 1 / (self.c + rank + 1)", "fused[i] = 1 / (self.c + rank + 1)"),
    ("byoai/rag.py", "if titles is not None and names and not names & set(words(titles[n - 1])):", "if False:"),
    ("byoai/agent.py", "if key in seen:", "if False:"),
    ("byoai/agent.py", "out = out[: self.max_result_chars]", "out = out"),
    ("byoai/agent.py", "if not target.is_relative_to(base):", "if False:"),
    ("byoai/agent.py", "if isinstance(node.op, ast.Pow) and abs(right) > 100:", "if False:"),
    ("byoai/agent.py", "if len(calls) >= self.max_calls:", "if False:"),
    ("byoai/planner.py", "for t in todo]", "for t in todo[:1]]"),
    ("byoai/mcp.py", "version = asked if asked in SUPPORTED_VERSIONS else SUPPORTED_VERSIONS[0]", "version = PROTOCOL_VERSION"),
    ("byoai/mcp.py", 'if "id" not in message:  # a notification', "if False:"),
    ("byoai/mcp.py", "if isinstance(message, list):", "if False:"),
    ("byoai/llm.py", "if (e.code == 429 or e.code >= 500) and attempt < self.retries:", "if attempt < self.retries:"),
    ("byoai/llm.py", "args = None", "args = {}"),
]


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        copy = Path(tmp) / "repo"
        shutil.copytree(ROOT, copy, ignore=shutil.ignore_patterns(".venv", ".git", "checkpoints", "__pycache__", ".pytest_cache"))
        env = {**os.environ, "PYTHONPATH": str(copy), "BYOAI_CHECKPOINTS": str(Path(tmp) / "checkpoints")}
        def run_suite() -> subprocess.CompletedProcess:
            return subprocess.run([sys.executable, "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider", "tests"], cwd=copy, env=env, capture_output=True, timeout=300)

        # The unmutated suite must pass first: otherwise every mutant would look "caught".
        baseline = run_suite()
        if baseline.returncode != 0:
            print("The test suite fails without any mutant; fix that first:\n" + baseline.stdout.decode(errors="replace")[-2000:])
            return 2
        survived = []
        for i, (file, original, mutant) in enumerate(MUTANTS, start=1):
            path = copy / file
            source = path.read_text(encoding="utf-8")
            if original not in source:
                print(f"{i:>2}. NOT FOUND in {file}: {original[:60]} (the code changed; update this list)")
                survived.append(file)
                continue
            path.write_text(source.replace(original, mutant, 1), encoding="utf-8")
            try:
                result = run_suite()
                caught = result.returncode != 0
            except subprocess.TimeoutExpired:
                caught = True  # a hang is a failure too
            path.write_text(source, encoding="utf-8")
            print(f"{i:>2}. {'caught  ' if caught else 'SURVIVED'}  {file}: {original[:70]}", flush=True)
            if not caught:
                survived.append(file)
        print(f"\n{len(MUTANTS) - len(survived)} of {len(MUTANTS)} mutants caught")
        return 1 if survived else 0


if __name__ == "__main__":
    sys.exit(main())
