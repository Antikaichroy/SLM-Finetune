"""
Step 3: Inference
==================
Load the fine-tuned Gemma model and answer shipping questions.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_PATH  = "./gemma_finetuned"   # or a checkpoint dir
MAX_NEW_TOK = 256
TEMPERATURE = 0.3
device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model     = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )
    model.eval()
    return tokenizer, model


def answer(question: str, tokenizer, model) -> str:
    prompt = f"<start_of_turn>user\n{question}<end_of_turn>\n<start_of_turn>model\n"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOK,
            temperature=TEMPERATURE,
            do_sample=TEMPERATURE > 0,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode only the newly generated tokens
    new_tokens = output_ids[0][inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


if __name__ == "__main__":
    print("Loading model...")
    tokenizer, model = load_model()
    print("Ready.\n")

    while True:
        q = input("Question (or 'quit'): ").strip()
        if q.lower() in ("quit", "exit", "q"):
            break
        if not q:
            continue
        resp = answer(q, tokenizer, model)
        print(f"\nAnswer: {resp}\n")
