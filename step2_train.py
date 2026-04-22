"""
Step 2: Fine-tune Gemma 270M with Grouped Weighted Loss
=========================================================
- Reads augmented_groups.jsonl produced by step1_augment.py
- For each group of 6 (q,a) pairs:
    * Runs all 6 forward passes
    * Computes weighted loss:
        L = 0.55 * L_ground_truth + 0.45 * mean(L_soft_1..5)
    * Single backward + single optimizer step per group
- Uses bf16, gradient checkpointing — fits comfortably in 24GB

Loss weighting rationale:
    Ground truth  : 0.55  (slightly higher — real label)
    Each soft label: 0.09  (0.45 / 5 — LLM-generated, lower trust)
    Total         : 1.00
"""

import json
import math
import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, get_cosine_schedule_with_warmup
from torch.optim import AdamW
from tqdm import tqdm

# ── Config ────────────────────────────────────────────────────────────────────
AUGMENTED_JSONL    = "augmented_groups.jsonl"
MODEL_NAME         = "google/gemma-3-1b-it"   # closest public HF id for Gemma ~270M
                                                # swap to "google/gemma-2-2b" etc as needed
OUTPUT_DIR         = "./gemma_finetuned"
MAX_SEQ_LEN        = 512

# Loss weights
GT_WEIGHT          = 0.55
SOFT_WEIGHT        = 0.45                      # split evenly across 5 soft labels

# Training
EPOCHS             = 3
LEARNING_RATE      = 2e-5
WARMUP_RATIO       = 0.05
GRAD_CLIP          = 1.0
SAVE_EVERY_N_STEPS = 200
LOG_EVERY_N_STEPS  = 20
SEED               = 42

# Memory
USE_BF16           = True                      # A100/3090/4090 — safe; else set False
GRAD_CHECKPOINTING = True
# ─────────────────────────────────────────────────────────────────────────────


def set_seed(seed: int):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ── Dataset ───────────────────────────────────────────────────────────────────

class ShippingGroupDataset(Dataset):
    """
    Each item is one GROUP: a list of 6 dicts, each with keys:
        question, answer, is_ground_truth
    """
    def __init__(self, jsonl_path: str):
        self.groups = []
        with open(jsonl_path, "r") as f:
            for line in f:
                obj = json.loads(line.strip())
                group = [obj["anchor"]] + obj["soft"]   # anchor first, then 5 soft
                self.groups.append(group)

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, idx):
        return self.groups[idx]   # list of 6 dicts


def collate_groups(batch):
    """
    batch: list of groups (each group = list of 6 dicts)
    Returns list of groups as-is — tokenization happens inside the training loop
    so we can handle variable lengths per pair cleanly.
    """
    return batch


# ── Tokenization helper ───────────────────────────────────────────────────────

def tokenize_qa_pair(question: str, answer: str, tokenizer, max_len: int, device):
    """
    Format as instruction-style prompt and tokenize.
    Labels are set to -100 for the prompt portion (only train on answer).
    """
    prompt = f"<start_of_turn>user\n{question}<end_of_turn>\n<start_of_turn>model\n"
    full   = prompt + answer + tokenizer.eos_token

    prompt_ids = tokenizer.encode(prompt, add_special_tokens=True)
    full_ids   = tokenizer.encode(full,   add_special_tokens=True, max_length=max_len,
                                  truncation=True)

    labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]

    # Align lengths (truncation may affect full_ids)
    min_len = min(len(full_ids), len(labels))
    input_ids = full_ids[:min_len]
    labels    = labels[:min_len]

    input_ids_t = torch.tensor(input_ids, dtype=torch.long, device=device).unsqueeze(0)
    labels_t    = torch.tensor(labels,    dtype=torch.long, device=device).unsqueeze(0)
    attn_mask   = torch.ones_like(input_ids_t)

    return input_ids_t, attn_mask, labels_t


# ── Loss for one (q, a) pair ──────────────────────────────────────────────────

def compute_pair_loss(model, input_ids, attention_mask, labels):
    """Standard causal LM loss, ignoring -100 positions."""
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    return outputs.loss


# ── Training ──────────────────────────────────────────────────────────────────

def train():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype  = torch.bfloat16 if (USE_BF16 and torch.cuda.is_available()) else torch.float32

    print(f"Device : {device}")
    print(f"Dtype  : {dtype}")

    # ── Load tokenizer & model ────────────────────────────────────────────
    print(f"\nLoading {MODEL_NAME} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=dtype,
        device_map="auto",          # auto-places layers across available GPUs
    )

    if GRAD_CHECKPOINTING:
        model.gradient_checkpointing_enable()

    model.train()

    # ── Dataset & loader ──────────────────────────────────────────────────
    dataset = ShippingGroupDataset(AUGMENTED_JSONL)
    loader  = DataLoader(dataset, batch_size=1, shuffle=True,
                         collate_fn=collate_groups)

    total_steps  = len(loader) * EPOCHS
    warmup_steps = max(1, int(total_steps * WARMUP_RATIO))

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    global_step  = 0
    running_loss = 0.0
    per_soft_weight = SOFT_WEIGHT / 5.0   # 0.09 per soft label

    print(f"\nGroups      : {len(dataset)}")
    print(f"Epochs      : {EPOCHS}")
    print(f"Total steps : {total_steps}")
    print(f"GT weight   : {GT_WEIGHT}  |  Soft weight per label: {per_soft_weight:.4f}\n")

    # ── Training loop ─────────────────────────────────────────────────────
    for epoch in range(EPOCHS):
        epoch_loss = 0.0

        for batch in tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            # batch is a list of 1 group (batch_size=1), each group = list of 6 dicts
            group = batch[0]   # list of 6 {"question", "answer", "is_ground_truth"}

            optimizer.zero_grad()

            group_loss = torch.tensor(0.0, device=device, dtype=torch.float32)

            # ── Forward pass for all 6 pairs ──────────────────────────────
            for pair in group:
                input_ids, attn_mask, labels = tokenize_qa_pair(
                    pair["question"], pair["answer"],
                    tokenizer, MAX_SEQ_LEN, device
                )

                # Cast to model dtype
                with torch.amp.autocast("cuda", dtype=dtype, enabled=torch.cuda.is_available()):
                    pair_loss = compute_pair_loss(model, input_ids, attn_mask, labels)

                weight = GT_WEIGHT if pair["is_ground_truth"] else per_soft_weight
                group_loss = group_loss + weight * pair_loss.float()

            # ── Single backward + single optimizer step per group ─────────
            group_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            scheduler.step()

            loss_val     = group_loss.item()
            running_loss += loss_val
            epoch_loss   += loss_val
            global_step  += 1

            # Logging
            if global_step % LOG_EVERY_N_STEPS == 0:
                avg = running_loss / LOG_EVERY_N_STEPS
                lr  = scheduler.get_last_lr()[0]
                print(f"  Step {global_step:5d} | loss {avg:.4f} | lr {lr:.2e}")
                running_loss = 0.0

            # Checkpoint
            if global_step % SAVE_EVERY_N_STEPS == 0:
                ckpt = os.path.join(OUTPUT_DIR, f"checkpoint-step{global_step}")
                model.save_pretrained(ckpt)
                tokenizer.save_pretrained(ckpt)
                print(f"  Saved checkpoint → {ckpt}")

        avg_epoch = epoch_loss / len(loader)
        print(f"\nEpoch {epoch+1} complete | avg loss: {avg_epoch:.4f}\n")

    # ── Final save ────────────────────────────────────────────────────────
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"\nTraining complete. Model saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    train()
