"""
Script de reprise du sweep hyperparamètres, tourne sur Colab (GPU).

UTILISATION :
1. Upload ce fichier dans : /content/drive/MyDrive/Colab Notebooks/sweep.py
   et définis les identifiants du jeu :
       import os; os.environ['MILLIONAIRE_USER'] = '...'; os.environ['MILLIONAIRE_PASS'] = '...'
2. Dans une cellule Colab, monte d'abord le Drive :
       from google.colab import drive
       drive.mount('/content/drive')
3. Lance en arrière-plan :
       import subprocess
       proc = subprocess.Popen(
           ["python", "/content/drive/MyDrive/Colab Notebooks/sweep.py"],
           stdout=open("/tmp/sweep.log", "w"),
           stderr=subprocess.STDOUT,
       )
       print("PID:", proc.pid)
4. Lis les logs en temps réel :
       !tail -f /tmp/sweep.log
5. Pour stopper : !kill <PID>
"""

import sys, os, csv, statistics
from itertools import product

# Accès à millionaire_client depuis Drive
sys.path.append('/content/drive/MyDrive/Colab Notebooks')
from millionaire_client import MillionaireClient

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, logging as hf_logging
hf_logging.set_verbosity_error()

# ── Config ────────────────────────────────────────────────────────────────────
API_URL   = "http://131.175.15.22:51111/"
USERNAME  = os.environ["MILLIONAIRE_USER"]   # set before running, never hard-code credentials
PASSWORD  = os.environ["MILLIONAIRE_PASS"]
CSV_PATH  = "/content/drive/MyDrive/Colab Notebooks/sweep_results.csv"

NUM_QUESTIONS    = 15
N_ATTEMPTS_SWEEP = 5
TEST_NUMBERS     = [0, 1, 2, 3, 4, 5]
categories_names = [
    "Entertainment", "Ancient History and Politics",
    "Science and Nature", "Maths",
    "Philosophy and Psychology", "News",
]

temperatures  = [0.7, 1.0, 1.3, 1.7, 2.0]
top_ps        = [0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
beams_options = [1, 2, 3]

# ── Prompts ───────────────────────────────────────────────────────────────────
PROMPT_FEW_SHOT = (
    "You are a quiz expert. Answer the multiple choice question.\n"
    "Choose ONLY the correct letter : A, B, C, or D.\n"
    "Do not explain.\n\n"
    "Example 1:\nQuestion: What is the capital of France?\n"
    "A) Berlin\nB) Madrid\nC) Paris\nD) Rome\nAnswer: C\n\n"
    "Example 2:\nQuestion: Who wrote Romeo and Juliet?\n"
    "A) Charles Dickens\nB) William Shakespeare\nC) Victor Hugo\nD) Mark Twain\nAnswer: B\n\n"
    "Now answer:"
)

# ── Connexion ─────────────────────────────────────────────────────────────────
print("Connexion au serveur...", flush=True)
client = MillionaireClient(API_URL)
client.login(USERNAME, PASSWORD)
print("Connected.", flush=True)

# ── Chargement modèle (GPU via device_map='auto') ─────────────────────────────
print("Chargement Mistral-7B-Instruct-v0.2...", flush=True)
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-Instruct-v0.2",
    torch_dtype=torch.float16,
    device_map="auto",
)
print("Modèle chargé.", flush=True)

# ── Fonctions ─────────────────────────────────────────────────────────────────
def ask_model(question, options, do_sample, temperature, top_p, num_beams):
    prompt = (
        f"{PROMPT_FEW_SHOT}\n\n"
        f"Question: {question}\n"
        f"A) {options[0]}\nB) {options[1]}\nC) {options[2]}\nD) {options[3]}\n"
        f"Answer:"
    )
    inputs       = tokenizer(prompt, return_tensors="pt").to("cuda")
    input_length = inputs.input_ids.shape[1]
    gen_kw = dict(
        **inputs,
        do_sample=do_sample,
        repetition_penalty=1.0,
        max_new_tokens=5,
        num_beams=num_beams,
    )
    if do_sample:
        if temperature: gen_kw["temperature"] = temperature
        if top_p:       gen_kw["top_p"]       = top_p
    outputs  = model.generate(**gen_kw)
    response = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True).strip()
    for c in response:
        if c in ("A", "B", "C", "D"):
            return c
    return "A"


def play_game(game, do_sample, temperature, top_p, num_beams):
    score   = 0
    mapping = {"A": 0, "B": 1, "C": 2, "D": 3}
    while game.in_progress:
        q = game.current_question
        if not q:
            break
        options = [opt.text for opt in q.options]
        try:
            ans = ask_model(q.text, options, do_sample, temperature, top_p, num_beams)
        except Exception as e:
            print(f"  [INFERENCE ERROR] {e}", flush=True)
            ans = "A"
        try:
            result = game.answer(mapping[ans])
        except Exception as e:
            print(f"  [CONNECTION ERROR] {e}", flush=True)
            break
        if result.correct:
            score += 1
        else:
            break
    return score


def run_grading(do_sample, temperature, top_p, num_beams):
    results = {i: [] for i in TEST_NUMBERS}
    for comp_id in TEST_NUMBERS:
        for k in range(N_ATTEMPTS_SWEEP):
            print(
                f"  Cat {comp_id} ({categories_names[comp_id]}) "
                f"| Tentative {k+1}/{N_ATTEMPTS_SWEEP}",
                flush=True,
            )
            try:
                game  = client.game.start(competition_id=comp_id)
                score = play_game(game, do_sample, temperature, top_p, num_beams)
            except Exception as e:
                print(f"  [GAME ERROR] {e}", flush=True)
                score = 0
            results[comp_id].append(score)
    return results


def save_row(row: dict):
    file_exists = os.path.exists(CSV_PATH)
    with open(CSV_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def load_done_keys():
    if not os.path.exists(CSV_PATH):
        return set()
    done = set()
    with open(CSV_PATH, newline="") as f:
        for row in csv.DictReader(f):
            done.add((float(row["temperature"]), float(row["top_p"]), int(row["num_beams"])))
    return done


# ── Sweep avec reprise automatique ────────────────────────────────────────────
done_keys = load_done_keys()
remaining = [
    (t, p, b)
    for t, p, b in product(temperatures, top_ps, beams_options)
    if (t, p, b) not in done_keys
]
total = len(temperatures) * len(top_ps) * len(beams_options)
print(f"{len(done_keys)}/{total} configs déjà faites, {len(remaining)} restantes.", flush=True)

for temp, top_p_val, n_beams in remaining:
    use_sampling = temp > 0
    print(f"\n>>> temp={temp} top_p={top_p_val} beams={n_beams}", flush=True)

    r = run_grading(
        do_sample=use_sampling,
        temperature=temp if use_sampling else None,
        top_p=top_p_val if use_sampling else None,
        num_beams=n_beams,
    )

    all_scores = [s for sl in r.values() for s in sl]
    total_s    = sum(all_scores)
    acc        = total_s / (N_ATTEMPTS_SWEEP * NUM_QUESTIONS * len(TEST_NUMBERS))

    row = {
        "temperature": temp,
        "top_p":       top_p_val,
        "num_beams":   n_beams,
        "total_score": total_s,
        "overall_acc": round(acc, 4),
        "mean_score":  round(statistics.mean(all_scores), 4),
        "std_score":   round(statistics.stdev(all_scores) if len(all_scores) > 1 else 0, 4),
        "min_score":   min(all_scores),
        "max_score":   max(all_scores),
        **{
            f"acc_{categories_names[i][:5]}":
            round(statistics.mean(r[i]) / NUM_QUESTIONS, 4)
            for i in TEST_NUMBERS
        },
    }
    save_row(row)
    print(f"  Saved -> acc={acc:.1%}", flush=True)

print("\nSweep terminé.", flush=True)
