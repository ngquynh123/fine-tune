#!/usr/bin/env python3
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE_ID = "vinai/PhoGPT-4B"          # hoặc PhoGPT-4B-Chat
ADAPTER = "/home/nguyen-quynh/Desktop/fine-tune/output-phogpt-lora"

tok = AutoTokenizer.from_pretrained(BASE_ID, use_fast=True, trust_remote_code=True)

# GTX 1650: dùng fp16 + device_map="auto"
base = AutoModelForCausalLM.from_pretrained(
    BASE_ID, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True
)
model = PeftModel.from_pretrained(base, ADAPTER)
model.eval()

def ask(instr, context=None, max_new_tokens=200):
    if context:
        prompt = f"### HƯỚNG DẪN:\n{instr}\n\n### NGỮ CẢNH:\n{context}\n\n### TRẢ LỜI:\n"
    else:
        prompt = f"### HƯỚNG DẪN:\n{instr}\n\n### TRẢ LỜI:\n"

    inputs = tok(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # đổi True nếu muốn random
            temperature=0.7,
            top_p=0.9
        )
    print(tok.decode(out[0], skip_special_tokens=True))

if __name__ == "__main__":
    # Ví dụ câu hỏi (đổi theo domain của bạn)
    ask("Khói trắng thường xuất hiện khi nào?"),
    ask("Quy tắc vượt xe máy trên đường cao tốc là gì?")
