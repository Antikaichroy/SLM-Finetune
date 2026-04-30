"""
Step 1: Augmentation Pipeline
==============================
For each row in your CSV (question, answer):
  - Uses Ollama (Gemma 3 8B) to generate 5 paraphrased questions
  - Uses Ollama to generate soft-label answers for each paraphrase
  - Saves a JSONL where each line is one GROUP of 6 (q,a) pairs

Expected CSV format:
    question,answer
    "What is the delivery status of order #123?","The order is in transit..."

Output: augmented_groups.jsonl
Each line:
{
  "group_id": 0,
  "anchor": {"question": "...", "answer": "...", "is_ground_truth": true},
  "soft": [
    {"question": "...", "answer": "...", "is_ground_truth": false},
    ...  (5 items)
  ]
}
"""

import json
import time
import pandas as pd
import requests
from tqdm import tqdm

# ── Config ────────────────────────────────────────────────────────────────────
CSV_PATH          = "shipping_data.csv"       # <-- change to your CSV path
OUTPUT_JSONL      = "augmented_groups.jsonl"
OLLAMA_URL        = "http://localhost:11434/api/generate"
ANCHOR_MODEL      = "gemma3:8b"
NUM_PARAPHRASES   = 5
TEMPERATURE       = 0.7                        # for paraphrase diversity
ANSWER_TEMPERATURE = 0.3                       # lower = more consistent answers
REQUEST_TIMEOUT   = 120
RETRY_ATTEMPTS    = 3
RETRY_DELAY       = 5                          # seconds between retries
# ─────────────────────────────────────────────────────────────────────────────


def ollama_generate(prompt: str, temperature: float = 0.7) -> str:
    """Call Ollama and return the response text."""
    payload = {
        "model": ANCHOR_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": 512,
        }
    }
    for attempt in range(RETRY_ATTEMPTS):
        try:
            resp = requests.post(OLLAMA_URL, json=payload, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
            return resp.json()["response"].strip()
        except Exception as e:
            if attempt < RETRY_ATTEMPTS - 1:
                print(f"  [retry {attempt+1}] Ollama error: {e}")
                time.sleep(RETRY_DELAY)
            else:
                print(f"  [failed] Skipping after {RETRY_ATTEMPTS} attempts: {e}")
                return ""


def generate_paraphrases(question: str) -> list[str]:
    """Ask Gemma to rewrite the question in 5 different ways."""
    prompt = f"""You are a data augmentation assistant for a shipping and logistics domain.

Your task: rewrite the following question in exactly {NUM_PARAPHRASES} different ways.
- Keep the exact same meaning and intent
- Vary the phrasing, sentence structure, and wording
- Each variant should feel natural, as if a real user typed it
- Output ONLY a numbered list (1. ... 2. ... etc.), nothing else

Original question: {question}

{NUM_PARAPHRASES} paraphrased versions:"""

    raw = ollama_generate(prompt, temperature=TEMPERATURE)

    # Parse numbered list
    lines = [l.strip() for l in raw.split("\n") if l.strip()]
    paraphrases = []
    for line in lines:
        # Strip leading "1." / "1)" / "-" etc.
        for prefix in ["1.", "2.", "3.", "4.", "5.", "6.",
                       "1)", "2)", "3)", "4)", "5)", "6)", "-", "*"]:
            if line.startswith(prefix):
                line = line[len(prefix):].strip()
                break
        if line:
            paraphrases.append(line)

    # Fallback: if parsing failed, duplicate with note
    while len(paraphrases) < NUM_PARAPHRASES:
        paraphrases.append(question)

    return paraphrases[:NUM_PARAPHRASES]


def generate_soft_answer(question: str) -> str:
    """Ask Gemma to answer the paraphrased shipping question."""
    prompt = f"""You are an expert shipping and logistics assistant.

Answer the following question clearly and concisely based on general shipping domain knowledge.
Be factual and specific. Do not add disclaimers or preamble.

Question: {question}

Answer:"""

    return ollama_generate(prompt, temperature=ANSWER_TEMPERATURE)


def check_ollama_running() -> bool:
    try:
        resp = requests.get("http://localhost:11434", timeout=5)
        return True
    except Exception:
        return False


def main():
    # ── Sanity checks ──────────────────────────────────────────────────────
    if not check_ollama_running():
        raise RuntimeError(
            "Ollama is not running. Start it with: ollama serve\n"
            f"Then pull the model: ollama pull {ANCHOR_MODEL}"
        )

    df = pd.read_csv(CSV_PATH)
    assert "question" in df.columns and "answer" in df.columns, \
        "CSV must have 'question' and 'answer' columns"

    print(f"Loaded {len(df)} rows from {CSV_PATH}")
    print(f"Anchor model: {ANCHOR_MODEL}")
    print(f"Output: {OUTPUT_JSONL}\n")

    # ── Resume support: skip already-processed group_ids ──────────────────
    processed_ids = set()
    try:
        with open(OUTPUT_JSONL, "r") as f:
            for line in f:
                obj = json.loads(line)
                processed_ids.add(obj["group_id"])
        print(f"Resuming — {len(processed_ids)} groups already done.\n")
    except FileNotFoundError:
        pass

    # ── Main loop ──────────────────────────────────────────────────────────
    with open(OUTPUT_JSONL, "a") as out_f:
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Augmenting"):
            if idx in processed_ids:
                continue

            question = str(row["question"]).strip()
            gt_answer = str(row["answer"]).strip()

            # 1. Generate 5 paraphrased questions
            paraphrases = generate_paraphrases(question)

            # 2. Generate soft-label answers for each paraphrase
            soft_pairs = []
            for para_q in paraphrases:
                soft_a = generate_soft_answer(para_q)
                soft_pairs.append({
                    "question": para_q,
                    "answer": soft_a,
                    "is_ground_truth": False
                })

            # 3. Build group record
            group = {
                "group_id": int(idx),
                "anchor": {
                    "question": question,
                    "answer": gt_answer,
                    "is_ground_truth": True
                },
                "soft": soft_pairs
            }

            out_f.write(json.dumps(group) + "\n")
            out_f.flush()  # safe resume on crash

    print(f"\nDone! Saved to {OUTPUT_JSONL}")
    print(f"Total groups: {len(df)}, each with 1 ground truth + {NUM_PARAPHRASES} soft pairs")


if __name__ == "__main__":
    main()
