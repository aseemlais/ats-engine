"""
Phase 3: Fine-Tuning Qwen2.5-1.5B-Instruct using Unsloth + QLoRA
and Exporting to GGUF Q4_K_M for lightweight CPU/RAM asynchronous ATS serving.

Can be run locally (with CUDA GPU) or in Google Colab / Kaggle.
"""

import argparse
import os
import sys
import json
import torch
from datasets import load_dataset


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune Qwen2.5-1.5B-Instruct for ATS resume extraction")
    parser.add_argument("--model-name", type=str, default="unsloth/Qwen2.5-1.5B-Instruct", help="Base model checkpoint")
    parser.add_argument("--max-seq-length", type=int, default=4096, help="Maximum context sequence length")
    parser.add_argument("--train-file", type=str, default="data/sft/train.jsonl", help="Path to train JSONL")
    parser.add_argument("--val-file", type=str, default="data/sft/val.jsonl", help="Path to validation JSONL")
    parser.add_argument("--output-dir", type=str, default="outputs/qwen2.5-1.5b-ats", help="Output directory for adapters")
    parser.add_argument("--gguf-dir", type=str, default="outputs/qwen2.5-1.5b-ats-gguf", help="Directory for exported GGUF model")
    parser.add_argument("--quant-method", type=str, default="q4_k_m", help="GGUF quantization method (e.g. q4_k_m, q8_0, f16)")
    parser.add_argument("--learning-rate", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=2, help="Per device batch size")
    parser.add_argument("--grad-accum", type=int, default=8, help="Gradient accumulation steps (effective batch size = batch_size * grad_accum)")
    parser.add_argument("--hf-repo", type=str, default=None, help="Hugging Face repo to push GGUF (e.g. username/qwen2.5-1.5b-ats-gguf)")
    parser.add_argument("--hf-token", type=str, default=os.getenv("HF_TOKEN"), help="Hugging Face write token")
    return parser.parse_args()


def main():
    args = parse_args()
    print("=" * 60)
    print("Starting Qwen2.5-1.5B ATS Resume Extractor Fine-Tuning")
    print(f"Base Model:       {args.model_name}")
    print(f"Max Seq Length:   {args.max_seq_length}")
    print(f"Train File:       {args.train_file}")
    print(f"Val File:         {args.val_file}")
    print(f"Effective Batch:  {args.batch_size * args.grad_accum}")
    print(f"Learning Rate:    {args.learning_rate}")
    print(f"Epochs:           {args.epochs}")
    print("=" * 60)

    try:
        from unsloth import FastLanguageModel
        from unsloth.chat_templates import get_chat_template, train_on_responses_only
        from trl import SFTTrainer
        from transformers import TrainingArguments
    except ImportError:
        print("\n[ERROR] Unsloth and required training libraries are not installed in this environment.")
        print("Please install them with:")
        print("  pip install \"unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git\"")
        print("  pip install --no-deps \"xformers<0.0.28\" trl peft accelerate bitsandbytes")
        sys.exit(1)

    # 1. Load Base Model in 4-bit
    print("\n[Step 1/5] Loading Qwen2.5-1.5B-Instruct in 4-bit quantization...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        dtype=None,  # Auto-detects FP16 on T4, BF16 on Ampere/Hopper
        load_in_4bit=True,
    )

    # 2. Add LoRA Adapters
    print("\n[Step 2/5] Configuring LoRA adapters (r=16, alpha=32, all linear projections)...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        use_gradient_checkpointing="unsloth",  # Unsloth 30% less VRAM checkpointing
        random_state=42,
        use_rslora=False,
        loftq_config=None,
    )

    # 3. Setup Qwen2.5 Chat Template & Format Dataset
    print("\n[Step 3/5] Setting up Qwen2.5 chat template and loading dataset...")
    tokenizer = get_chat_template(
        tokenizer,
        chat_template="qwen-2.5",
    )

    def formatting_prompts_func(examples):
        convos = examples["messages"]
        texts = [
            tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False)
            for convo in convos
        ]
        return {"text": texts}

    dataset_dict = {}
    if os.path.exists(args.train_file):
        train_raw = load_dataset("json", data_files=args.train_file, split="train")
        train_dataset = train_raw.map(formatting_prompts_func, batched=True)
        print(f"Loaded {len(train_dataset)} training examples.")
    else:
        raise FileNotFoundError(f"Train file not found: {args.train_file}")

    val_dataset = None
    if os.path.exists(args.val_file):
        val_raw = load_dataset("json", data_files=args.val_file, split="train")
        val_dataset = val_raw.map(formatting_prompts_func, batched=True)
        print(f"Loaded {len(val_dataset)} validation examples.")

    # 4. Training Loop with Completion-Only Loss Masking
    print("\n[Step 4/5] Initializing SFTTrainer with completion-only loss masking...")
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        warmup_ratio=0.03,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=10,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        seed=42,
        save_strategy="epoch",
        save_total_limit=2,
        evaluation_strategy="epoch" if val_dataset else "no",
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        dataset_num_proc=2,
        packing=False,
        args=training_args,
    )

    # Compute loss ONLY on the assistant's JSON output (masks user prompt tokens)
    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|im_start|>user\n",
        response_part="<|im_start|>assistant\n",
    )

    print("\nStarting training loop...")
    trainer_stats = trainer.train()
    print(f"\nTraining completed! Train loss: {trainer_stats.training_loss:.4f}")

    # Save LoRA Adapter
    print(f"Saving LoRA adapters to {args.output_dir}...")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # 5. GGUF Export (Q4_K_M)
    print(f"\n[Step 5/5] Exporting merged model to GGUF format ({args.quant_method})...")
    os.makedirs(args.gguf_dir, exist_ok=True)
    model.save_pretrained_gguf(
        args.gguf_dir,
        tokenizer,
        quantization_method=args.quant_method,
    )
    print(f"GGUF model successfully saved to: {args.gguf_dir}")

    # Optional Push to Hugging Face Hub
    if args.hf_repo:
        print(f"\nPushing GGUF model to Hugging Face Hub: {args.hf_repo}...")
        model.push_to_hub_gguf(
            args.hf_repo,
            tokenizer,
            quantization_method=args.quant_method,
            token=args.hf_token,
        )
        print(f"Successfully pushed GGUF model to https://huggingface.co/{args.hf_repo}")

    print("\n=== Phase 3 Training & Export Completed Successfully! ===")


if __name__ == "__main__":
    main()
