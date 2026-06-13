import logging
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

from src.llm_platform.training.config import SFTConfig

logger = logging.getLogger(__name__)

class LoRAForgeTrainer:
    def __init__(self, config: SFTConfig):
        self.config = config
        self.tokenizer = None
        self.model = None

    def setup_model(self):
        logger.info(f"Loading tokenizer for {self.config.model.model_name_or_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model.model_name_or_path,
            token=self.config.model.hf_token,
            trust_remote_code=True
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"

        logger.info("Configuring BitsAndBytes for 4-bit QLoRA")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=self.config.model.use_4bit,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=self.config.model.use_nested_quant,
            bnb_4bit_compute_dtype=torch.bfloat16
        )

        logger.info("Loading base model")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model.model_name_or_path,
            quantization_config=bnb_config,
            device_map="auto",
            token=self.config.model.hf_token
        )
        self.model.config.use_cache = False 
        
        self.model = prepare_model_for_kbit_training(self.model)

        logger.info("Injecting LoRA adapters")
        peft_config = LoraConfig(
            r=self.config.lora.r,
            lora_alpha=self.config.lora.lora_alpha,
            lora_dropout=self.config.lora.lora_dropout,
            target_modules=self.config.lora.target_modules,
            bias="none",
            task_type="CAUSAL_LM"
        )
        self.model = get_peft_model(self.model, peft_config)
        self.model.print_trainable_parameters()

    def prepare_dataset(self):
        logger.info("Loading datasets")
        dataset = load_dataset("json", data_files={
            "train": self.config.data.train_file,
            "test": self.config.data.test_file
        })
        
        def formatting_prompts_func(example):
            output_texts = []
            for i in range(len(example['messages'])):
                text = self.tokenizer.apply_chat_template(example["messages"][i], tokenize=False)
                output_texts.append(text)
            return output_texts
        response_template = "<|im_start|>assistant\n"
        collator = DataCollatorForCompletionOnlyLM(response_template=response_template, tokenizer=self.tokenizer)

        return dataset, formatting_prompts_func, collator

    def train(self):
        """Сборка пайплайна и запуск обучения."""
        self.setup_model()
        dataset, formatter, collator = self.prepare_dataset()

        training_args = TrainingArguments(
            output_dir=self.config.training.output_dir,
            per_device_train_batch_size=self.config.training.per_device_train_batch_size,
            gradient_accumulation_steps=self.config.training.gradient_accumulation_steps,
            learning_rate=self.config.training.learning_rate,
            num_train_epochs=self.config.training.num_train_epochs,
            logging_steps=self.config.training.logging_steps,
            save_steps=self.config.training.save_steps,
            report_to=self.config.training.report_to,
            gradient_checkpointing=True,         # Экономит ~30% VRAM
            optim="paged_adamw_32bit",           # Выгружает состояния оптимизатора в CPU RAM при пиках
            bf16=True,                           # Включаем bfloat16
            max_grad_norm=0.3,                   # Обрезка градиентов от "взрывов"
            warmup_ratio=0.03,                   # Плавный разогрев LR
            lr_scheduler_type="cosine",          # Косинусное затухание LR
        )

        logger.info("Initializing SFTTrainer")
        trainer = SFTTrainer(
            model=self.model,
            train_dataset=dataset["train"],
            eval_dataset=dataset["test"],
            args=training_args,
            formatting_func=formatter,
            data_collator=collator,
            max_seq_length=self.config.data.max_seq_length,
        )

        logger.info(">>> Starting Training Loop <<<")
        trainer.train()
        
        logger.info(f"✅ Training complete. Saving adapters to {self.config.training.output_dir}")
        trainer.model.save_pretrained(self.config.training.output_dir)
        self.tokenizer.save_pretrained(self.config.training.output_dir)