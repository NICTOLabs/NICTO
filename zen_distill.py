"""
NICTO FAST Pipeline — Groq-powered distillation
Groq is fastest (100+ tokens/sec), fully free, no reasoning overhead.
"""
import sys, os, time, json
from pathlib import Path
import urllib.request

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).parent))

def log(msg):
    ts = time.strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open("zen_distill.log", "a") as f:
        f.write(line + "\n")

def load_keys():
    keys = {}
    with open(".env") as f:
        for line in f:
            if "=" in line and not line.startswith("#"):
                k, v = line.strip().split("=", 1)
                keys[k] = v
    return keys

SYSTEM = "You are a helpful, knowledgeable assistant. Provide clear and detailed responses."

def call_groq(api_key, model, prompt):
    body = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": prompt[:500]},
        ],
        "max_tokens": 400,
        "temperature": 0.7,
    }).encode()
    h = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    for _ in range(2):
        try:
            req = urllib.request.Request("https://api.groq.com/openai/v1/chat/completions", data=body, headers=h)
            resp = urllib.request.urlopen(req, timeout=30)
            r = json.loads(resp.read())
            return r["choices"][0]["message"]["content"].strip()
        except:
            time.sleep(0.5)
    return None

def call_or(api_key, model, prompt):
    body = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": prompt[:500]},
        ],
        "max_tokens": 400,
        "temperature": 0.7,
    }).encode()
    h = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json", "HTTP-Referer": "https://github.com/NICTOLabs/NICTO", "X-Title": "NICTO"}
    for _ in range(2):
        try:
            req = urllib.request.Request("https://openrouter.ai/api/v1/chat/completions", data=body, headers=h)
            resp = urllib.request.urlopen(req, timeout=30)
            r = json.loads(resp.read())
            msg = r["choices"][0]["message"]
            return (msg.get("content") or msg.get("reasoning_content") or "").strip()
        except:
            time.sleep(0.5)
    return None

def load_prompts(n=3000):
    import numpy as np
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file("nicto_ai/tokenizer/artifacts/tokenizer.json")
    prompts = []
    for bf in sorted(Path("training_data").glob("*.bin")):
        if bf.stat().st_size == 0: continue
        data = np.memmap(str(bf), dtype=np.uint16, mode="r")
        for i in range(0, min(len(data), n * 150), 150):
            if len(prompts) >= n: break
            chunk = data[i:i+200]
            if len(chunk) > 30:
                text = tok.decode(chunk.tolist()).strip()
                if len(text) > 40:
                    prompts.append(text[:300])
        if len(prompts) >= n: break
    return prompts[:n]

def gen_teachers(keys):
    out = Path("teacher_outputs.jsonl")
    if out.exists() and sum(1 for _ in open(out, encoding="utf-8")) > 1000:
        c = sum(1 for _ in open(out, encoding="utf-8"))
        log(f"Teachers exist: {c}")
        return c

    prompts = load_prompts(3000)
    log(f"Generating from {len(prompts)} prompts via Groq + OpenRouter")

    groq_models = ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"]
    or_models = ["nvidia/nemotron-3-ultra-550b-a55b:free", "tencent/hy3:free", "google/gemma-4-31b-it:free"]

    t0 = time.time()
    count = 0
    with open(out, "w", encoding="utf-8") as f:
        for i, prompt in enumerate(prompts):
            if i % 5 == 0:
                model = or_models[(i // 5) % len(or_models)]
                resp = call_or(keys["OPENROUTER_API_KEY"], model, prompt)
                name = model.split("/")[-1].replace(":free", "")
            else:
                model = groq_models[i % len(groq_models)]
                resp = call_groq(keys["GROQ_API_KEY"], model, prompt)
                name = model

            if resp and len(resp) > 20:
                f.write(json.dumps({"prompt": prompt, "response": resp, "teacher": name}, ensure_ascii=False) + "\n")
                f.flush()
                count += 1

            if (i + 1) % 100 == 0:
                elapsed = time.time() - t0
                speed = (i+1) / max(elapsed, 0.1)
                log(f"  {i+1}/{len(prompts)} | {count} ok | {speed:.1f}/s | {(len(prompts)-i-1)/max(speed,0.01)/60:.0f}m left")
            time.sleep(0.1)

    log(f"Generated {count} in {time.time()-t0:.0f}s")
    return count

def create_bin():
    import numpy as np
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file("nicto_ai/tokenizer/artifacts/tokenizer.json")
    out = Path("training_data/zen_distill.bin")
    if out.exists() and out.stat().st_size > 10000:
        log(f"Bin exists: {out.stat().st_size//2/1e6:.1f}M tokens")
        return
    f = open(out, "wb")
    total = 0
    for line in open("teacher_outputs.jsonl", encoding="utf-8"):
        rec = json.loads(line)
        if len(rec["response"]) < 20: continue
        ids = tok.encode(rec["response"]).ids
        if len(ids) > 2048: ids = ids[:2048]
        np.array(ids + [2], dtype=np.uint16).tofile(f)
        total += len(ids) + 1
    f.close()
    log(f"Bin: {total/1e6:.1f}M tokens")

def run(cmd, label):
    import subprocess
    log(f">>> {label}")
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=43200)
    log(f"  Exit: {r.returncode}")
    return r.returncode

def find_ckpt(d, ft=False):
    p = Path(d)
    if ft:
        for n in ["best_ft.pt", "final_ft.pt"]:
            if (p/n).exists(): return str(p/n)
    for n in ["best.pt", "final.pt"]:
        if (p/n).exists(): return str(p/n)
    return str(sorted(p.glob("step_*.pt"))[-1]) if list(p.glob("step_*.pt")) else None

def main():
    t0 = time.time()
    keys = load_keys()
    log("="*50)
    log("NICTO FAST DISTILLATION")
    log(f"Start: {time.strftime('%H:%M:%S')}")
    log("="*50)

    gen_teachers(keys)
    create_bin()

    # Train medium (2000 steps)
    run([sys.executable, "train_master.py", "--config", "medium",
         "--steps", "2000", "--seq-len", "512", "--batch-size", "1",
         "--lr", "5e-4", "--warmup", "100", "--save-every", "500", "--log-every", "50"],
        "Train medium 42.8M")

    teacher = find_ckpt("checkpoints_master/medium")
    if teacher:
        run([sys.executable, "distill.py", "--teacher", teacher,
             "--config-student", "200m", "--steps", "1500",
             "--seq-len", "512", "--batch-size", "1", "--lr", "3e-4",
             "--warmup", "50", "--temperature", "2.0", "--alpha", "0.5",
             "--save-every", "500", "--log-every", "25"],
            "Distill -> 200M")

    student = find_ckpt("checkpoints_master/200m")
    if student:
        run([sys.executable, "finetune.py", "--checkpoint", student,
             "--steps", "500", "--seq-len", "512", "--batch-size", "1",
             "--lr", "1e-4", "--warmup", "25", "--save-every", "250",
             "--log-every", "25"],
            "Fine-tune 200M")

    final = find_ckpt("checkpoints_master/200m", ft=True) or find_ckpt("checkpoints_master/200m")
    if final:
        run([sys.executable, "validate_model.py", "--checkpoint", final], "Validate")

    log(f"\nDONE: {(time.time()-t0)/60:.0f}m | Model: {final}")

if __name__ == "__main__":
    main()
