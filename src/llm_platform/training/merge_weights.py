import logging
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from src.llm_platform.training.config import SFTConfig

logger = logging.getLogger(__name__)

class WeightMerger:
    def __init__(self, config: SFTConfig):
        self.config = config

    def merge_and_export(self, export_path: str):
        logger.info(f"Loading base model: {self.config.model.model_name_or_path} in BF16")
        
        base_model = AutoModelForCausalLM.from_pretrained(
            self.config.model.model_name_or_path,
            torch_dtype=torch.bfloat16,
            device_map="cpu",
            token=self.config.model.hf_token
        )

        logger.info(f"Loading LoRA adapters from: {self.config.training.output_dir}")
        peft_model = PeftModel.from_pretrained(
            base_model,
            self.config.training.output_dir,
            torch_dtype=torch.bfloat16,
        )

        logger.info("Merging weights (This might take a few minutes...)")
        merged_model = peft_model.merge_and_unload()

        logger.info(f"Saving merged monolithic model to {export_path}")
        merged_model.save_pretrained(export_path)
        
        tokenizer = AutoTokenizer.from_pretrained(self.config.training.output_dir)
        tokenizer.save_pretrained(export_path)
        logger.info("✅ Merge completed successfully!")