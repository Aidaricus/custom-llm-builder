# src/llm_platform/training/config.py
from typing import List, Optional
from pydantic import BaseModel, Field
import yaml
from pathlib import Path

class ModelConfig(BaseModel):
    model_name_or_path: str = Field(..., description="HuggingFace ID (e.g., 'Qwen/Qwen2.5-1.5B') or local path")
    use_4bit: bool = Field(default=True, description="Enable QLoRA 4-bit quantization")
    use_nested_quant: bool = Field(default=True, description="Enable double quantization for QLoRA")
    hf_token: Optional[str] = Field(default=None, description="HuggingFace token for gated models (like Llama-3)")

class DataConfig(BaseModel):
    train_file: str = Field(..., description="Path to formatted train.jsonl")
    test_file: str = Field(..., description="Path to formatted test.jsonl")
    max_seq_length: int = Field(default=2048, description="Maximum context length")

class LoRAConfigParams(BaseModel):
    r: int = Field(default=16, description="LoRA attention dimension (rank)")
    lora_alpha: int = Field(default=32, description="LoRA alpha parameter")
    lora_dropout: float = Field(default=0.05, description="LoRA dropout probability")
    target_modules: List[str] = Field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        description="Modules to apply LoRA adapters"
    )

class TrainingConfigParams(BaseModel):
    output_dir: str = Field(default="results/sft_model", description="Directory to save checkpoints")
    learning_rate: float = Field(default=2e-4, description="Learning rate")
    per_device_train_batch_size: int = Field(default=2, description="Batch size per GPU")
    gradient_accumulation_steps: int = Field(default=4, description="Number of updates steps to accumulate")
    num_train_epochs: int = Field(default=3, description="Total number of training epochs")
    logging_steps: int = Field(default=10, description="Log metrics every X steps")
    save_steps: int = Field(default=100, description="Save checkpoint every X steps")
    report_to: str = Field(default="tensorboard", description="Integration to report metrics (e.g., 'wandb', 'tensorboard')")
    
    optim: str = Field(default="paged_adamw_32bit", description="Optimizer type")
    bf16: bool = Field(default=True, description="Use bfloat16 precision")
    fp16: bool = Field(default=False, description="Use fp16 precision (if bf16 is not supported)")
    max_grad_norm: float = Field(default=0.3, description="Max gradient norm")
    warmup_ratio: float = Field(default=0.03, description="Linear warmup over warmup_ratio fraction of total steps")
    lr_scheduler_type: str = Field(default="cosine", description="Learning rate scheduler type")
    gradient_checkpointing: bool = Field(default=True, description="Enable gradient checkpointing to save memory")

class SFTConfig(BaseModel):
    """Main configuration class that holds all sub-configs."""
    model: ModelConfig
    data: DataConfig
    lora: LoRAConfigParams
    training: TrainingConfigParams

    @classmethod
    def load_from_yaml(cls, yaml_path: str | Path) -> "SFTConfig":
        """Loads configuration from a YAML file."""
        with open(yaml_path, "r", encoding="utf-8") as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)