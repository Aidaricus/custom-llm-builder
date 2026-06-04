import json
import logging
import re
from pathlib import Path
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


class DatasetSanitizer:
    """
    Cleans and deduplicates generated datasets before formatting.
    """
    def __init__(self, similarity_threshold: float = 0.85):
        self.similarity_threshold = similarity_threshold
        
        # Regex-паттерны для отлова типичных галлюцинаций, извинений и лени ИИ
        self.refusal_patterns = [
            r"(?i)as an ai",
            r"(?i)i am an ai",
            r"(?i)i cannot fulfill",
            r"(?i)i'm sorry, but",
            r"(?i)извините, но",
            r"(?i)как искусственный интеллект",
            r"(?i)я не могу (помочь|создать|выполнить|ответить)",
            r"(?i)я являюсь (языковой моделью|ии)",
            r"(?i)however, it is important to note",
            r"(?i)важно отметить, что",
            r"(?i)я всего лишь"
        ]
        self.compiled_patterns = [re.compile(p) for p in self.refusal_patterns]

    def _has_refusal(self, text: str) -> bool:
        """Проверяет текст на наличие типичных маркеров отказа."""
        return any(pattern.search(text) for pattern in self.compiled_patterns)

    def _is_valid_length(self, text: str, min_words: int = 5, max_words: int = 2000) -> bool:
        """Отсеивает слишком короткие отписки и слишком длинные 'зацикленные' ответы."""
        word_count = len(text.split())
        return min_words <= word_count <= max_words

    def _calculate_jaccard(self, text1: str, text2: str) -> float:
        """
        Легкая проверка на смысловое сходство без тяжелых ML-библиотек.
        Считает пересечение уникальных слов.
        """
        set1 = set(text1.lower().split())
        set2 = set(text2.lower().split())
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        return intersection / union if union > 0 else 0.0

    def sanitize_sft(self, input_file: str | Path, output_file: str | Path) -> Dict[str, int]:
        """
        Очищает SFT датасет (сырые SFTPair JSONL).
        Возвращает словарь со статистикой очистки.
        """
        input_file = Path(input_file)
        output_file = Path(output_file)
        
        seen_prompts: List[str] = []
        stats = {
            "total_processed": 0,
            "passed": 0,
            "rejected_refusal": 0,
            "rejected_length": 0,
            "rejected_duplicate": 0,
            "rejected_structure": 0,
        }

        valid_pairs = []

        if not input_file.exists():
            logger.error(f"Input file not found: {input_file}")
            return stats

        with open(input_file, "r", encoding="utf-8") as f:
            for line in f:
                stats["total_processed"] += 1
                try:
                    pair = json.loads(line)
                    messages = pair.get("messages", [])
                    
                    user_msg = next((m["content"] for m in messages if m["role"] == "user"), "")
                    assistant_msg = next((m["content"] for m in messages if m["role"] == "assistant"), "")

                    if not user_msg or not assistant_msg:
                        stats["rejected_structure"] += 1
                        continue

                    # 1. Проверка длины ответа
                    if not self._is_valid_length(assistant_msg):
                        stats["rejected_length"] += 1
                        continue

                    # 2. Проверка на ИИ-отказы и воду
                    if self._has_refusal(assistant_msg):
                        stats["rejected_refusal"] += 1
                        continue

                    # 3. Семантическая дедупликация (сравниваем вопросы)
                    is_duplicate = False
                    for seen in seen_prompts:
                        if self._calculate_jaccard(user_msg, seen) >= self.similarity_threshold:
                            is_duplicate = True
                            break
                    
                    if is_duplicate:
                        stats["rejected_duplicate"] += 1
                        continue

                    # Если все проверки пройдены, сохраняем пару
                    seen_prompts.append(user_msg)
                    valid_pairs.append(pair)
                    stats["passed"] += 1

                except json.JSONDecodeError:
                    stats["rejected_structure"] += 1
                    continue
        
        # Сохраняем очищенный датасет
        with open(output_file, "w", encoding="utf-8") as f:
            for pair in valid_pairs:
                f.write(json.dumps(pair, ensure_ascii=False) + "\n")

        logger.info(f"SFT Sanitization complete. Stats: {stats}")
        return stats
    
    def sanitize_dpo(self, input_file: str | Path, output_file: str | Path) -> Dict[str, int]:
        """
        Очищает DPO датасет (сырые DPOTriplet JSONL).
        Отсеивает триплеты, где rejected-ответ содержит отказ или слишком похож на chosen.
        """
        input_file = Path(input_file)
        output_file = Path(output_file)
        
        stats = {
            "total_processed": 0,
            "passed": 0,
            "rejected_structure": 0,
            "rejected_refusal": 0,
            "rejected_similarity": 0,
            "rejected_length": 0
        }

        valid_triplets = []

        if not input_file.exists():
            logger.error(f"Input file not found: {input_file}")
            return stats

        with open(input_file, "r", encoding="utf-8") as f:
            for line in f:
                stats["total_processed"] += 1
                try:
                    triplet = json.loads(line)
                    prompt = triplet.get("prompt", "")
                    chosen = triplet.get("chosen", "")
                    rejected = triplet.get("rejected", "")

                    if not prompt or not chosen or not rejected:
                        stats["rejected_structure"] += 1
                        continue

                    # 1. Проверка длины rejected-ответа
                    if not self._is_valid_length(rejected):
                        stats["rejected_length"] += 1
                        continue

                    # 2. Проверка на ИИ-отказы (плохой ответ не должен быть "Я не знаю")
                    if self._has_refusal(rejected):
                        stats["rejected_refusal"] += 1
                        continue

                    # 3. Защита от совпадений (Similarity Check)
                    # Если плохой ответ почти идентичен хорошему - это плохой контраст для модели
                    similarity = self._calculate_jaccard(chosen, rejected)
                    if similarity >= self.similarity_threshold:
                        stats["rejected_similarity"] += 1
                        continue

                    # Все проверки пройдены
                    valid_triplets.append(triplet)
                    stats["passed"] += 1

                except json.JSONDecodeError:
                    stats["rejected_structure"] += 1
                    continue
        
        with open(output_file, "w", encoding="utf-8") as f:
            for triplet in valid_triplets:
                f.write(json.dumps(triplet, ensure_ascii=False) + "\n")

        logger.info(f"DPO Sanitization complete. Stats: {stats}")
        return stats