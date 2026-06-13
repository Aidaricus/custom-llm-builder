import json
import logging
import random
from pathlib import Path
from typing import List, Dict, Any
from .adapters import get_adapter

logger = logging.getLogger(__name__)

class DatasetFormatter:
    """
    Utility class to convert internal Pydantic schemas into standard 
    HuggingFace/Axolotl formats and split them into training and validation sets.
    """
    def __init__(self, seed: int = 42):
        self.seed = seed
        random.seed(self.seed)

    def _split_and_save(self, data: List[Dict[str, Any]], train_file: Path, test_file: Path, train_ratio: float):
        """Helper method to shuffle, split, and save data to JSONL."""
        random.shuffle(data)
        split_idx = int(len(data) * train_ratio)
        
        train_data = data[:split_idx]
        test_data = data[split_idx:]

        for file_path, split_data in [(train_file, train_data), (test_file, test_data)]:
            with open(file_path, "w", encoding="utf-8") as f:
                for item in split_data:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
                    
        logger.info(f"Saved {len(train_data)} train rows to {train_file.name}")
        logger.info(f"Saved {len(test_data)} test rows to {test_file.name}")

    def format_sft(
            self,
            input_file: str | Path,
            train_file: str | Path,
            test_file: str | Path,
            train_ratio: float = 0.9,
            input_format: str = "native"
    ):
        """Formats SFT data to standard 'messages' format and splits it."""
        input_file = Path(input_file)
        if not input_file.exists():
            logger.error(f"Input file not found: {input_file}")
            return

        try:
            adapter = get_adapter(input_format)
        except ValueError as e:
            logger.error(e)
            return

        formatted_data = []
        with open(input_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    raw_data = json.loads(line)
                    adapted_row = adapter(raw_data)
                    # For SFT, we only need the conversation history
                    formatted_data.append({"messages": raw_data.get("messages", [])})
                    if adapted_row.get("messages"):
                        formatted_data.append(adapted_row)
                except json.JSONDecodeError:
                    logger.warning("Failed to decode JSON line, skipping.")
                    continue

        self._split_and_save(formatted_data, Path(train_file), Path(test_file), train_ratio)

    def format_dpo(self, input_file: str | Path, train_file: str | Path, test_file: str | Path, train_ratio: float = 0.9):
        """Formats DPO data into standard TRL conversational format and splits it."""
        input_file = Path(input_file)
        if not input_file.exists():
            logger.error(f"Input file not found: {input_file}")
            return

        formatted_data = []
        with open(input_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    raw_data = json.loads(line)
                    prompt = raw_data.get("prompt", "")
                    chosen_text = raw_data.get("chosen", "")
                    rejected_text = raw_data.get("rejected", "")
                    
                    # Constructing the conversational DPO format
                    formatted_row = {
                        "prompt": prompt,
                        "chosen": [
                            {"role": "user", "content": prompt},
                            {"role": "assistant", "content": chosen_text}
                        ],
                        "rejected": [
                            {"role": "user", "content": prompt},
                            {"role": "assistant", "content": rejected_text}
                        ]
                    }
                    formatted_data.append(formatted_row)
                except json.JSONDecodeError:
                    continue

        self._split_and_save(formatted_data, Path(train_file), Path(test_file), train_ratio)