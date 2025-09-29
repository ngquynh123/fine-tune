#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, json
import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    set_seed,
)
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

# ========== CONFIG ==========
MODEL_ID = os.environ.get("PHO_MODEL", "vinai/PhoGPT-4B")   # hoặc "vinai/PhoGPT-4B-Chat"
OUT_DIR  = os.environ.get("OUT_DIR",  "output-phogpt-lora")
SEED     = int(os.environ.get("SEED", "1337"))

# Dữ liệu
TRAIN_PATH = os.environ.get("TRAIN_PATH", "data/train.jsonl")
VAL_PATH   = os.environ.get("VAL_PATH",   "data/val.jsonl")

# Token/seq (4GB-friendly)
MAX_SEQ_LEN = int(os.environ.get("MAX_SEQ_LEN", "96"))  # khuyên 96; nếu còn OOM, thử 80

# LoRA
LORA_R          = int(os.environ.get("LORA_R", "8"))     # giảm để tiết kiệm VRAM
LORA_ALPHA      = int(os.environ.get("LORA_ALPHA", "32"))
LORA_DROPOUT    = float(os.environ.get("LORA_DROPOUT", "0.05"))
LORA_TARGET_ENV = os.environ.get("LORA_TARGET", "Wqkv,out_proj").strip()  # rút gọn targets

# Train
EPOCHS        = float(os.environ.get("EPOCHS", "1"))
LR            = float(os.environ.get("LR", "1e-5"))
BATCH_SIZE    = int(os.environ.get("BATCH_SIZE", "1"))      # 4GB: để 1
GRAD_ACC      = int(os.environ.get("GRAD_ACC", "32"))       # bù batch nhỏ
WARMUP_RATIO  = float(os.environ.get("WARMUP_RATIO", "0.05"))
LOG_STEPS     = int(os.environ.get("LOG_STEPS", "10"))

USE_4BIT = os.environ.get("USE_4BIT", "1") == "1"


def load_jsonl(path: str) -> Dataset:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                rows.append(json.loads(s))
    return Dataset.from_list(rows)


def format_example(ex):
    """Format {instruction,input,output} -> prompt kiểu Instruct VN; bỏ mẫu rỗng."""
    instruction = (ex.get("instruction") or "").strip()
    xinput      = (ex.get("input") or "").strip()
    output      = (ex.get("output") or "").strip()

    if not instruction or not output:
        return ""  # loại bỏ

    if xinput:
        prompt = (
            "### HƯỚNG DẪN:\n"
            f"{instruction}\n\n"
            "### NGỮ CẢNH:\n"
            f"{xinput}\n\n"
            "### TRẢ LỜI:\n"
        )
    else:
        prompt = "### HƯỚNG DẪN:\n" + instruction + "\n\n### TRẢ LỜI:\n"

    return prompt + output


def pick_lora_targets(model) -> list:
    """Chọn target_modules dựa theo kiến trúc + lọc theo module tồn tại."""
    if LORA_TARGET_ENV:
        return [t for t in LORA_TARGET_ENV.split(",") if t]

    mtype = (getattr(model.config, "model_type", "") or "").lower()
    cls   = model.__class__.__name__.lower()
    if "mpt" in mtype or "mpt" in cls:
        candidates = ["Wqkv", "wqkv", "out_proj", "up_proj", "down_proj"]
    elif "bloom" in mtype or "bloom" in cls:
        candidates = ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"]
    else:
        candidates = ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]

    present = set()
    for name, _ in model.named_modules():
        for c in candidates:
            if name.endswith(c):
                present.add(c)
    return list(present) if present else candidates


def main():
    set_seed(SEED)

    # ---- Tokenizer ----
    tok = AutoTokenizer.from_pretrained(
        MODEL_ID, use_fast=True, trust_remote_code=True
    )
    tok.add_special_tokens({
        "bos_token": tok.bos_token or "<s>",
        "eos_token": tok.eos_token or "</s>",
        "unk_token": tok.unk_token or "<unk>",
        "pad_token": tok.pad_token or "<pad>",
    })
    tok.padding_side = "right"

    # ---- Model ----
    if USE_4BIT:
        from transformers import BitsAndBytesConfig
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,  # GTX 1650 dùng fp16
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        base = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            quantization_config=bnb_cfg,
            torch_dtype=torch.float16,
            device_map="auto",
            max_memory={0: "3500MiB", "cpu": "16GiB"},  # ép offload khi cần để tránh OOM
            trust_remote_code=True,
        )
        base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=False)
    else:
        base = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True,
        )

    base.resize_token_embeddings(len(tok))
    if hasattr(base.config, "use_cache"):
        base.config.use_cache = False

    # ---- LoRA ----
    target_modules = pick_lora_targets(base)
    lora_cfg = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=target_modules,
    )
    model = get_peft_model(base, lora_cfg)

    # --- Kiểm tra tham số trainable ---
    trainable, total = 0, 0
    for p in model.parameters():
        n = p.numel()
        total += n
        if p.requires_grad:
            trainable += n
    pct = (100.0 * trainable / total) if total else 0.0
    print(f"[LoRA] trainable params: {trainable:,} / {total:,} ({pct:.4f}%)")
    print(f"[LoRA] target_modules: {target_modules}")

    # ---- Data ----
    ds_train = load_jsonl(TRAIN_PATH)
    ds_eval  = load_jsonl(VAL_PATH)

    def map_fn(example):
        return {"text": format_example(example)}

    ds_train = ds_train.map(map_fn, batched=False, remove_columns=ds_train.column_names)
    ds_eval  = ds_eval.map(map_fn,  batched=False, remove_columns=ds_eval.column_names)

    # Lọc chuỗi trống (tránh loss=0)
    ds_train = ds_train.filter(lambda ex: ex["text"].strip() != "")
    ds_eval  = ds_eval.filter(lambda ex: ex["text"].strip() != "")

    print("[Data] train size:", len(ds_train))
    print("[Data] val size:", len(ds_eval))
    if len(ds_train) > 0:
        print("[Sample]", ds_train[0]["text"][:300].replace("\n", "\\n"))

    # ---- SFT Config ----
    sft_cfg = SFTConfig(
        output_dir=OUT_DIR,
        max_seq_length=MAX_SEQ_LEN,
        packing=False,
        dataset_text_field="text",

        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=max(1, BATCH_SIZE),
        gradient_accumulation_steps=GRAD_ACC,
        num_train_epochs=EPOCHS,
        learning_rate=LR,
        warmup_ratio=WARMUP_RATIO,
        lr_scheduler_type="cosine",

        fp16=True, bf16=False,
        max_grad_norm=1.0,
        optim="adamw_bnb_8bit",     # 8-bit optimizer để giảm VRAM

        logging_steps=LOG_STEPS,
        evaluation_strategy="no",        # tắt eval trong train để nhẹ VRAM
        save_strategy="no",         # nếu muốn resume, đổi thành "steps" và set save_steps
        save_total_limit=1,

        gradient_checkpointing=False,  # MPT hay lỗi khi bật
        dataloader_num_workers=0,
        report_to="none",
        run_name="phogpt-lora",
    )

    # ---- Trainer ----
    trainer = SFTTrainer(
        model=model,
        args=sft_cfg,
        train_dataset=ds_train,
        eval_dataset=None,   # tắt eval trong train cho nhẹ VRAM
        tokenizer=tok,
    )

    # Sanity check 1 mẫu để chắc loss > 0
    if len(ds_train) > 0:
        batch = tok(ds_train[0]["text"], return_tensors="pt",
                    truncation=True, max_length=MAX_SEQ_LEN).to(model.device)
        with torch.no_grad():
            out = model(**batch, labels=batch["input_ids"])
        print("[Sanity] single-sample loss =", float(out.loss))

    trainer.train()
    trainer.save_model(OUT_DIR)
    tok.save_pretrained(OUT_DIR)
    print("✅ Done. LoRA adapter saved at:", OUT_DIR)


if __name__ == "__main__":
    # Khuyên set thêm:
    # export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    # export TOKENIZERS_PARALLELISM=false
    main()
