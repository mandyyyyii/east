from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer

from trl import (
    ModelConfig,
    ScriptArguments,
    SFTConfig,
    # SFTTrainer,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)
from sft_trainer import SFTTrainer

if __name__ == "__main__":
    parser = TrlParser((ScriptArguments, SFTConfig, ModelConfig))
    parser.add_argument(
        "--use_weighting",
        action="store_true",
        help="Enable weighting during training"
    )


    parser.add_argument(
        "--use_entropy_non_linear",
        type=float,
        default=0,
        help="Use entropy non linear value"
    )

    parser.add_argument(
        "--use_entropy_non_linear_coeff",
        type=float,
        default=0,
        help="Use entropy non linear coeff value"
    )

    parser.add_argument(
        "--use_chosen_weight_non_linear",
        type=float,
        default=0,
        help="Use chosen weight non linear value"
    )

    parser.add_argument(
        "--use_chosen_weight_non_linear_coeff",
        type=float,
        default=0,
        help="Use chosen weight non linear coeff value"
    )

    parser.add_argument(
        "--use_rejected_weight_non_linear",
        type=float,
        default=0,
        help="Use rejected weight non linear value"
    )

    parser.add_argument(
        "--use_rejected_weight_non_linear_coeff",
        type=float,
        default=0,
        help="Use rejected weight non linear coeff value"
    )

    parser.add_argument(
        "--data_seed_t",
        type=int,
        default=42,
        help="dataset seed"
    )
    script_args, training_args, model_config, extra_config = parser.parse_args_and_config()
    training_args.data_seed=extra_config.data_seed_t#fix the seed
    print('seed', training_args.seed, training_args.data_seed)
    for attr in dir(extra_config):
        if not attr.startswith("_"):
            setattr(training_args, attr, getattr(extra_config, attr))
    ################
    # Model init kwargs & Tokenizer
    ################
    quantization_config = get_quantization_config(model_config)
    model_kwargs = dict(
        revision=model_config.model_revision,
        trust_remote_code=model_config.trust_remote_code,
        attn_implementation=model_config.attn_implementation,
        torch_dtype=model_config.torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )
    training_args.model_init_kwargs = model_kwargs
    tokenizer = AutoTokenizer.from_pretrained(
        model_config.model_name_or_path, trust_remote_code=model_config.trust_remote_code, use_fast=True
    )
    tokenizer.pad_token = tokenizer.eos_token

    ################
    # Dataset
    ################
    dataset = load_dataset("json", data_files=script_args.dataset_name)
    
    # Create a DatasetDict with all data in train and empty test set
    dataset = DatasetDict({
        'train': dataset['train'],
        'test': dataset['train'].select(range(0))  # Creates empty dataset with same schema
    })
    def add_weights_to_dataset(example):
        return {
            'entropy': example['entropy'],
            'chosen_weight': example['chosen_weight'],
            'rejected_weight': example['rejected_weight']
        }

    # Add weights to your dataset
    dataset = dataset.map(add_weights_to_dataset)

    ################
    # Training
    ################
    trainer = SFTTrainer(
        model=model_config.model_name_or_path,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=dataset[script_args.dataset_test_split],
        processing_class=tokenizer,
        peft_config=get_peft_config(model_config),
    )

    trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)
