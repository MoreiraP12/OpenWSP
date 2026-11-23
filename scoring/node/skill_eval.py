"""Held-out skill evaluation of one model directory with vLLM (runs inside the vllm/vllm-openai image).

python skill_eval.py SKILL MODEL_DIR --split heldout|practice [--n N] [--kwargs JSON] --out OUT.json
"""
import argparse
import json

from vllm import LLM, SamplingParams

import skills


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("skill")
    ap.add_argument("model")
    ap.add_argument("--split", default="heldout")
    ap.add_argument("--n", type=int)
    ap.add_argument("--kwargs", default="{}")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    llm = LLM(a.model, gpu_memory_utilization=0.12, max_model_len=8192, seed=0)  # shares GPUs with a served solver
    open(a.out + ".up", "w").close()  # the engine holds its memory now: score_write.py lets the next scorer start one
    tok = llm.get_tokenizer()

    # a turn ends at the chat end token even when the checkpoint's generation config only stops at end-of-text
    # (a base model edited into a chat model would otherwise run on past its answer)
    ends = [i for t in ("<|im_end|>", "<|endoftext|>", "<end_of_turn>", "<|eot_id|>")
            if (i := tok.convert_tokens_to_ids(t)) is not None and i != tok.unk_token_id]

    def gen(prompts, max_tokens, temperature, stop):
        sp = SamplingParams(temperature=temperature, top_p=0.95 if temperature > 0 else 1.0, max_tokens=max_tokens, stop=stop,
                            stop_token_ids=ends)
        return [o.outputs[0].text for o in llm.generate(prompts, sp, use_tqdm=False)]
    kw = dict(json.loads(a.kwargs), **({"n": a.n} if a.n else {}))
    res = skills.run(a.skill, tok, gen, a.split, **kw)
    json.dump(res, open(a.out, "w"))
    print(json.dumps(res))


if __name__ == "__main__":
    main()
