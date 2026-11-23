"""General-text loss of skill runs' edited models next to the unedited model: retention.py JOBS_JSON [GPU]

JOBS_JSON: [{"spec": "specs/v6/<item>.json", "run_id": "<16 hex>" or null for the unedited model}, ...]
Each edited checkpoint is merged exactly as for scoring (score_write.merged_dir). Loss = mean token NLL in nats over 200
WikiText-103 test paragraphs (raw text, no chat template, at most 512 tokens each). Prints one JSON line per job.
A skill edit that keeps the model a language model leaves this close to the unedited value; replacing the network
with a task-specific program does not.
"""
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import score_write as S  # noqa: E402
import wspnode as W  # noqa: E402


def main():
    jobs = json.loads(Path(sys.argv[1]).read_text())
    gpu = sys.argv[2] if len(sys.argv) > 2 else "3"
    texts = S.wiki_texts("scored")
    for j in jobs:
        spec = json.loads((W.ROOT / j["spec"]).read_text())
        item = W.ROOT / "items" / spec["opaque"]
        run_dir = W.ROOT / "runs" / j["run_id"] if j.get("run_id") else None
        tmp = W.ROOT / "tmp" / f"ret_{j.get('run_id') or 'base_' + spec['id']}"
        try:
            S.merged_dir(item, "M1", run_dir, tmp)
            out = {"item": spec["id"], "run_id": j.get("run_id"), "nll": round(S.dir_nll(tmp, texts, f"cuda:{gpu}"), 4)}
        except Exception as e:  # noqa: BLE001
            out = {"item": spec["id"], "run_id": j.get("run_id"), "error": str(e)[:300]}
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        print(json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
