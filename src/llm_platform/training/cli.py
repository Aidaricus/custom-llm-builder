import argparse
import logging
import sys
from pathlib import Path

from src.llm_platform.training.config import SFTConfig
from src.llm_platform.training.sft_trainer import LoRAForgeTrainer
from src.llm_platform.training.merge_weights import WeightMerger

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("TrainingCLI")

def main():
    parser = argparse.ArgumentParser(description="LoRA Forge: SFT Training and Merging CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Команда: train (Запуск SFT)
    train_parser = subparsers.add_parser("train", help="Run Supervised Fine-Tuning (QLoRA)")
    train_parser.add_argument("--config", type=Path, required=True, help="Path to YAML configuration file")

    # Команда: merge (Слияние весов)
    merge_parser = subparsers.add_parser("merge", help="Merge LoRA adapters into base model")
    merge_parser.add_argument("--config", type=Path, required=True, help="Path to YAML configuration file")
    merge_parser.add_argument("--export-dir", type=Path, required=True, help="Directory to save the merged FP16 model")

    args = parser.parse_args()

    try:
        config = SFTConfig.load_from_yaml(args.config)

        if args.command == "train":
            logger.info("Initializing LoRA Forge Trainer...")
            trainer = LoRAForgeTrainer(config)
            trainer.train()

        elif args.command == "merge":
            logger.info("Initializing Weight Merger...")
            merger = WeightMerger(config)
            merger.merge_and_export(str(args.export_dir))

    except Exception as e:
        logger.error(f"Execution failed: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()