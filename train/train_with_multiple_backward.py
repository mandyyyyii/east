#!/usr/bin/env python3
"""
Training script for trainer4 with DeepSpeed Zero compatibility.
This script demonstrates how to train with multiple backward passes while being compatible with DeepSpeed.
"""

import os
import sys
import torch
import argparse
from dataclasses import dataclass, field
from typing import Optional

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    HfArgumentParser,
)
from datasets import load_dataset

# Import the enhanced DPO trainer
from dpo_trainer import DPOTrainer, DPOConfig


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to train.
    """
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )


@dataclass
class DataArguments:
    """
    Arguments pertaining to what data we are going to input our model for training.
    """
    dataset_name: str = field(
        default="Anthropic/hh-rlhf", metadata={"help": "The name of the dataset to use"}
    )
    max_length: int = field(default=512, metadata={"help": "Maximum sequence length"})


@dataclass
class TrainingArgumentsWithMultipleBackward(TrainingArguments):
    """
    Extended training arguments that include multiple backward pass options.
    """
    # Multiple backward pass configuration
    multiple_backward_passes: bool = field(
        default=False, 
        metadata={"help": "Enable multiple backward passes for trainer4 compatibility"}
    )
    
    # Secondary loss configuration
    secondary_loss_weight: float = field(
        default=0.1, 
        metadata={"help": "Weight for secondary loss component"}
    )
    
    # Regularization options
    l2_reg_weight: float = field(
        default=0.0, 
        metadata={"help": "L2 regularization weight"}
    )
    
    aux_loss_weight: float = field(
        default=0.0, 
        metadata={"help": "Auxiliary loss weight"}
    )
    
    # Consistency loss
    consistency_loss_enabled: bool = field(
        default=False, 
        metadata={"help": "Enable consistency loss computation"}
    )
    
    # For traditional multiple backward (non-DeepSpeed)
    enable_second_backward: bool = field(
        default=False, 
        metadata={"help": "Enable actual second backward pass when not using DeepSpeed"}
    )


def prepare_dataset(tokenizer, dataset_name, max_length):
    """
    Prepare the dataset for DPO training.
    """
    # Load dataset
    dataset = load_dataset(dataset_name, split="train[:1000]")  # Small subset for testing
    
    def tokenize_function(examples):
        # This is a simplified tokenization - adjust based on your dataset format
        chosen = examples.get("chosen", examples.get("response", ""))
        rejected = examples.get("rejected", examples.get("rejected_response", ""))
        prompt = examples.get("prompt", examples.get("question", ""))
        
        # Tokenize prompt, chosen, and rejected responses
        prompt_tokens = tokenizer(prompt, truncation=True, max_length=max_length//3)
        chosen_tokens = tokenizer(chosen, truncation=True, max_length=max_length//3)
        rejected_tokens = tokenizer(rejected, truncation=True, max_length=max_length//3)
        
        return {
            "prompt_input_ids": prompt_tokens["input_ids"],
            "chosen_input_ids": chosen_tokens["input_ids"],
            "rejected_input_ids": rejected_tokens["input_ids"],
        }
    
    tokenized_dataset = dataset.map(tokenize_function, batched=False)
    return tokenized_dataset


def main():
    # Parse arguments
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArgumentsWithMultipleBackward, DPOConfig))
    model_args, data_args, training_args, dpo_config = parser.parse_args_into_dataclasses()
    
    # Load model and tokenizer
    print(f"Loading model from {model_args.model_name_or_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        torch_dtype=torch.float16 if training_args.fp16 else torch.float32,
        device_map="auto" if not training_args.deepspeed else None,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name or model_args.model_name_or_path
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Prepare dataset
    print(f"Preparing dataset: {data_args.dataset_name}")
    train_dataset = prepare_dataset(tokenizer, data_args.dataset_name, data_args.max_length)
    
    # Initialize trainer
    trainer = DPOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        **dpo_config.__dict__
    )
    
    # Print configuration info
    print("\n" + "="*50)
    print("TRAINING CONFIGURATION")
    print("="*50)
    print(f"Model: {model_args.model_name_or_path}")
    print(f"Dataset: {data_args.dataset_name}")
    print(f"Multiple backward passes: {training_args.multiple_backward_passes}")
    print(f"DeepSpeed enabled: {training_args.deepspeed is not None}")
    if training_args.deepspeed:
        print(f"DeepSpeed config: {training_args.deepspeed}")
    print(f"Secondary loss weight: {training_args.secondary_loss_weight}")
    print(f"L2 regularization weight: {training_args.l2_reg_weight}")
    print(f"Gradient accumulation steps: {training_args.gradient_accumulation_steps}")
    print("="*50 + "\n")
    
    # Check for DeepSpeed compatibility
    if training_args.multiple_backward_passes and training_args.deepspeed:
        print("⚠️  Multiple backward passes requested with DeepSpeed.")
        print("   Using DeepSpeed-compatible combined loss approach.")
        print("   Multiple loss components will be combined into a single backward pass.\n")
    elif training_args.multiple_backward_passes:
        print("⚠️  Multiple backward passes requested without DeepSpeed.")
        print("   Using traditional multiple backward approach.\n")
    
    # Start training
    print("Starting training...")
    trainer.train()
    
    # Save model
    print("Saving model...")
    trainer.save_model()
    
    print("Training completed successfully!")


if __name__ == "__main__":
    main()