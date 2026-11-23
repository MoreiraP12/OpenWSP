"""Node-side session manager, called by the Mac driver over ssh.

  session.py start RUN_ID ITEM_ID ACCESS WRITE   create run dir + network-less kernel container
  session.py exec  RUN_ID                       run code from stdin in the kernel
  session.py fwd   RUN_ID                       forward-pass request JSON from stdin
  session.py stop  RUN_ID                       remove the container (run dir is kept)
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(os.environ.get("WSP_ROOT", "/ephemeral/wspbench"))
IMAGE = "wsp-kernel:v1"  # v0 plus the UTF-8 kernel.py/client.py (node/kernel/Dockerfile.v1); RL kernels run it too
PY = str(ROOT / "venv" / "bin" / "python")
CPUS = 16
EXEC_TIMEOUT = int(os.environ.get("WSP_EXEC_TIMEOUT", "300"))  # a long torch call can't be interrupted by the kernel's alarm, so the container is restarted


def cname(run_id):
    return "wsp_" + run_id.replace("/", "_")


def wait_kernel(run_id):
    for _ in range(60):
        if subprocess.run(["docker", "exec", cname(run_id), "test", "-S", "/tmp/kernel.sock"],
                          capture_output=True).returncode == 0:
            return
        time.sleep(1)


def kernel_stopped(run_id, run_dir):
    """If the run's kernel container has stopped (killed out of memory, say), start it again, log it to
    run_dir/kernel_restarts.jsonl and return True. A running container, or none at all, returns False."""
    for attempt in range(3):  # the daemon may still show the container running for a moment after its processes died
        r = subprocess.run(["docker", "inspect", "-f", "{{.State.Running}} {{.State.OOMKilled}} {{.State.ExitCode}}",
                            cname(run_id)], capture_output=True, text=True)
        if r.returncode != 0:
            return False
        if r.stdout.split()[:1] == ["false"]:
            break
        if attempt == 2:
            return False
        time.sleep(1)
    _, oom, code = r.stdout.split()
    subprocess.run(["docker", "start", cname(run_id)], capture_output=True)
    wait_kernel(run_id)
    with open(run_dir / "kernel_restarts.jsonl", "a") as f:
        f.write(json.dumps({"t": round(time.time()), "oom_killed": oom == "true", "exit_code": int(code)}) + "\n")
    return True


def allowed_gpus():
    """GPUs the benchmark's forward passes and scoring may use (never 0-2: teammates): WSP_GPUS, else ROOT/gpus.txt
    (set while a served solver fills the others), else 3-7."""
    f = ROOT / "gpus.txt"
    return [int(g) for g in (os.environ.get("WSP_GPUS") or (f.read_text().strip() if f.exists() else "3,4,5,6,7")).split(",")]


def free_gpu(need_mib=0, max_wait=0):
    """The GPU with the most free memory among the ones this benchmark may use (never 0-2: teammates); with need_mib,
    wait up to max_wait seconds for one that has that much free."""
    allowed = allowed_gpus()
    t0 = time.time()
    while True:
        q = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used,memory.total", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True).stdout
        free = {int(a): int(c) - int(b) for a, b, c in (l.split(",") for l in q.strip().splitlines())}
        g = max(allowed, key=lambda g: free.get(g, -1))
        if free.get(g, -1) >= need_mib or time.time() - t0 > max_wait:
            return str(g)
        time.sleep(10)


FWD_SLOTS = 32  # forward passes at once on the node: about 6 per GPU on GPUs 3-7 (v8). On one shared GPU, 30 unqueued
# ones ran it out of memory and 5 left it half idle, so v7 used 8 with gpus.txt = 3.
_slots = ROOT / "fwd_slots.txt"  # optional override, e.g. while a served solver shares the only free card
if _slots.exists():
    FWD_SLOTS = int(_slots.read_text().strip())


def fwd_slot():
    """Hold one of FWD_SLOTS node-wide locks (released when the process exits)."""
    import fcntl
    (ROOT / "locks").mkdir(exist_ok=True)
    while True:
        for i in range(FWD_SLOTS):
            f = open(ROOT / "locks" / f"fwd_{i}.lock", "w")
            try:
                fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
                return f
            except OSError:
                f.close()
        time.sleep(3)


def main():
    cmd, run_id = sys.argv[1], sys.argv[2]
    run_dir = ROOT / "runs" / run_id
    if cmd == "start":
        item_id, access, write = sys.argv[3], sys.argv[4], sys.argv[5] == "1"
        item_dir = ROOT / "items" / item_id
        (run_dir / "work").mkdir(parents=True, exist_ok=True)
        os.chmod(run_dir / "work", 0o777)
        slots = sorted(p.name for p in (item_dir / "public").iterdir() if p.is_dir())
        (run_dir / "run.json").write_text(json.dumps({"item_dir": str(item_dir), "access": access, "write": write,
                                                      "slots": slots, "passes": 0}, indent=1))
        subprocess.run(["docker", "rm", "-f", cname(run_id)], capture_output=True)
        r = subprocess.run(["docker", "run", "-d", "--name", cname(run_id), "--network", "none",
                            "--cpus", str(CPUS), "--memory", "96g", "--pids-limit", "512",
                            "--security-opt", "no-new-privileges", "--cap-drop", "ALL", "--user", "1003:1004",
                            "--read-only", "--tmpfs", "/tmp:rw,size=8g",
                            "-e", f"WSP_WRITE={int(write)}", "-e", f"WSP_TIMEOUT={EXEC_TIMEOUT - 30}",
                            # torch would otherwise start one thread per host core (208) inside a 16-CPU quota
                            "-e", f"OMP_NUM_THREADS={CPUS}", "-e", f"MKL_NUM_THREADS={CPUS}",
                            "-e", f"OPENBLAS_NUM_THREADS={CPUS}",
                            "-v", f"{item_dir / 'public'}:/model:ro", "-v", f"{run_dir / 'work'}:/work",
                            IMAGE], capture_output=True, text=True)
        wait_kernel(run_id)
        print(json.dumps({"ok": r.returncode == 0, "slots": slots, "err": r.stderr[-500:]}))
    elif cmd == "exec":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # the kernel's output is UTF-8, whatever the locale
        try:
            r = subprocess.run(["docker", "exec", "-i", cname(run_id), "python", "/kernel/client.py"],
                               input=sys.stdin.buffer.read().decode("utf-8", "replace"), capture_output=True,
                               encoding="utf-8", errors="replace", timeout=EXEC_TIMEOUT)
        except subprocess.TimeoutExpired:
            subprocess.run(["docker", "restart", "-t", "1", cname(run_id)], capture_output=True)
            wait_kernel(run_id)
            sys.stdout.write(f"TIMEOUT: the execution exceeded {EXEC_TIMEOUT} s. The kernel was restarted and ALL "
                             "variables were lost (files in /work are kept). Use cheaper computations: float32, "
                             "torch.linalg.svdvals or torch.svd_lowrank instead of full SVDs, fewer layers at once, "
                             "and cache intermediate results to /work.")
            return
        if r.returncode != 0 and kernel_stopped(run_id, run_dir):
            # Until Nov 22 a container that died (usually out of memory) stayed down and every later code run
            # returned "Error response from daemon: container ... is not running" (train/DECISIONS.md section 34).
            sys.stdout.write(r.stdout + "\nKERNEL DIED: the kernel's container stopped, most likely out of memory (its "
                             "limit is 96 GB). It was restarted and ALL variables were lost (files in /work are kept). "
                             "Use cheaper computations: float32 instead of float64, no vocabulary-by-vocabulary "
                             "products, large tensors in chunks, and cache intermediate results to /work.")
            return
        sys.stdout.write(r.stdout if r.returncode == 0 else r.stdout + "\n[kernel error] " + r.stderr[-2000:])
    elif cmd == "fwd":
        req, slot = sys.stdin.read(), fwd_slot()  # queue for a slot and for memory instead of failing out of memory
        for attempt in range(4):
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=free_gpu(10000, 1800), HF_HOME=str(ROOT / "hf_home"),
                       TRANSFORMERS_VERBOSITY="error", TOKENIZERS_PARALLELISM="false")
            r = subprocess.run([PY, str(ROOT / "forward.py"), str(run_dir)], input=req,
                               capture_output=True, text=True, env=env, timeout=3600)
            out = r.stdout.strip().splitlines()
            res = out[-1] if out else json.dumps({"ok": False, "error": "forward failed: " + r.stderr[-1500:]})
            if "out of memory" not in res.lower() or attempt == 3:
                break
            time.sleep(30)
        slot.close()
        print(res)
    elif cmd == "score":
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=free_gpu(10000, 1800), HF_HOME=str(ROOT / "hf_home"),
                   TRANSFORMERS_VERBOSITY="error", TOKENIZERS_PARALLELISM="false")
        spec = ROOT / "specs" / sys.argv[3] / f"{sys.argv[4]}.json"
        r = subprocess.run([PY, str(ROOT / "score_write.py"), str(run_dir), str(spec)], capture_output=True,
                           text=True, env=env, timeout=3600)
        out = r.stdout.strip().splitlines()
        print(out[-1] if out else json.dumps({"quality": 0.0, "error": r.stderr[-1500:]}))
    elif cmd == "stop":
        subprocess.run(["docker", "rm", "-f", cname(run_id)], capture_output=True)
        print(json.dumps({"ok": True}))


if __name__ == "__main__":
    main()
