# PhoGPT Fine-tuning with LoRA

Fine-tuning Vietnamese GPT model (PhoGPT) using LoRA (Low-Rank Adaptation) for efficient training.

## Features

- 🚀 **Memory-efficient**: Uses 4-bit quantization and LoRA for low VRAM usage
- 🇻🇳 **Vietnamese optimized**: Designed for Vietnamese instruction-following tasks  
- ⚙️ **Configurable**: All hyperparameters configurable via environment variables
- 🎯 **Instruction format**: Uses Vietnamese prompt format with "HƯỚNG DẪN", "NGỮ CẢNH", "TRẢ LỜI"

## Requirements

```bash
pip install torch transformers datasets peft trl bitsandbytes accelerate
```

## Data Format

Training data should be in JSONL format with the following structure:

```json
{"instruction": "Câu hỏi hoặc yêu cầu", "input": "Ngữ cảnh (tùy chọn)", "output": "Câu trả lời"}
```

## Usage

### Basic Usage
```bash
python fine-tune.py
```

### Custom Configuration
```bash
# Set environment variables
export EPOCHS=2
export BATCH_SIZE=2
export LR=2e-5
export MAX_SEQ_LEN=128
export LORA_R=16

python fine-tune.py
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PHO_MODEL` | `vinai/PhoGPT-4B` | Model to fine-tune |
| `OUT_DIR` | `output-phogpt-lora` | Output directory |
| `TRAIN_PATH` | `data/train.jsonl` | Training data path |
| `VAL_PATH` | `data/val.jsonl` | Validation data path |
| `MAX_SEQ_LEN` | `96` | Maximum sequence length |
| `LORA_R` | `8` | LoRA rank |
| `LORA_ALPHA` | `32` | LoRA alpha |
| `EPOCHS` | `1` | Number of epochs |
| `LR` | `1e-5` | Learning rate |
| `BATCH_SIZE` | `1` | Batch size per device |
| `GRAD_ACC` | `32` | Gradient accumulation steps |
| `USE_4BIT` | `1` | Enable 4-bit quantization |

## GPU Memory Requirements

- **4GB GPU**: Default settings (batch_size=1, max_seq_len=96)
- **8GB GPU**: Can increase batch_size=2, max_seq_len=128
- **CPU only**: Set `USE_4BIT=0`

## Files

- `fine-tune.py`: Main training script
- `infer.py`: Inference script (if exists)
- `data/`: Training and validation data
- `output-phogpt-lora/`: Model outputs

## Vietnamese Prompt Format

```
### HƯỚNG DẪN:
{instruction}

### NGỮ CẢNH:
{input}

### TRẢ LỜI:
{output}
```