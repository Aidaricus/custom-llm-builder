import asyncio
import json
import logging
import random
import uuid
from pathlib import Path
from typing import List, Optional, Dict
import yaml

from src.llm_platform.data_foundry.llm_client import LLMClient
from src.llm_platform.data_foundry.schemas import SFTPair, LLMGeneratedContent

logger = logging.getLogger(__name__)

class EvolPipeline:
    def __init__(
        self,
        input_file: str | Path,
        output_file: str | Path,
        prompts_file: str | Path,
        model_name: str = "google/gemma-4-31b-it:free",
        max_concurrent_requests: int = 3,
    ):
        self.input_file = Path(input_file)
        self.output_file = Path(output_file)
        self.prompts_file = Path(prompts_file)
        
        self.client = LLMClient(model_name=model_name)
        self.semaphore = asyncio.Semaphore(max_concurrent_requests)
        
        self.evol_prompts = self._load_prompts()

    def _load_prompts(self) -> Dict[str, str]:
        try:
            with open(self.prompts_file, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
                return data.get("evol_prompts", {})
        except Exception as e:
            logger.error(f"Failed to load prompts from {self.prompts_file}: {e}")
            return {}
    
    async def _process_single_pair(self, pair: SFTPair) -> Optional[SFTPair]:
        if not self.evol_prompts:
            logger.error("No evol prompts loaded.")
            return None

        original_q = next((m.content for m in pair.messages if m.role == "user"), "")
        original_a = next((m.content for m in pair.messages if m.role == "assistant"), "")
        
        if not original_q or not original_a:
            return None

        user_prompt = f"Оригинальный вопрос:\n{original_q}\n\nОригинальный ответ:\n{original_a}"

        mutation_name, base_system_prompt = random.choice(list(self.evol_prompts.items()))

        # ИСПРАВЛЕНИЕ 1: Усиленный системный промпт для защиты от галлюцинаций
        system_prompt_enhanced = (
            f"{base_system_prompt}\n\n"
            "CRITICAL RULES FOR OUTPUT:\n"
            "1. The 'user' role MUST contain ONLY the new, realistic question. NEVER include your instructions or meta-prompts (like 'Rewrite this' or 'Add a table') in the user message.\n"
            "2. The 'assistant' role MUST contain the detailed, accurate answer to the new question.\n"
            "3. Do NOT wrap the entire dialogue inside the 'assistant' role."
        )

        user_prompt = f"Original Question:\n{original_q}\n\nOriginal Answer:\n{original_a}\n\nGenerate the evolved User-Assistant pair."

        async with self.semaphore:
            try:
                llm_content = await self.client.generate_structured_data(
                    system_prompt=system_prompt_enhanced,
                    user_prompt=user_prompt,
                    response_model=LLMGeneratedContent
                )
                await asyncio.sleep(5)
                
                messages = llm_content.messages
                if len(messages) < 2:
                    raise ValueError("LLM generated fewer than 2 messages (missing user or assistant role).")
                
                has_user = any(m.role == "user" for m in messages)
                has_assistant = any(m.role == "assistant" for m in messages)
                
                if not (has_user and has_assistant):
                    raise ValueError("LLM output is missing either the 'user' or 'assistant' role.")
                
                evolved_pair = SFTPair(
                    pair_id=f"evol_{uuid.uuid4().hex[:8]}",
                    source_chunk_id=pair.source_chunk_id,
                    messages=llm_content.messages,
                    is_evolved=True
                )
                
                logger.debug(f"Pair {pair.pair_id} evolved using '{mutation_name}'")
                
                return evolved_pair
                
            except ValueError as ve:
                logger.warning(f"Validation failed for pair {pair.pair_id}: {ve}")
                return None
            except Exception as e:
                logger.error(f"Failed to evolve pair {pair.pair_id}: {e}")
                return None

    async def run_evolution(self):
        if not self.input_file.exists():
            logger.error(f"Input file not found: {self.input_file}")
            return

        base_pairs: List[SFTPair] = []
        with open(self.input_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    pair = SFTPair.model_validate_json(line)
                    if not pair.is_evolved:
                        base_pairs.append(pair)
                except Exception as e:
                    logger.warning(f"Skipping invalid line: {e}")

        logger.info(f"Loaded {len(base_pairs)} base pairs for evolution.")

        tasks = [self._process_single_pair(pair) for pair in base_pairs]
        results = await asyncio.gather(*tasks)

        valid_evolved_pairs: List[SFTPair] = [res for res in results if res is not None]

        logger.info(f"Writing {len(valid_evolved_pairs)} evolved pairs to {self.output_file}...")
        with open(self.output_file, "w", encoding="utf-8") as f:
            for pair in valid_evolved_pairs:
                f.write(pair.model_dump_json() + "\n")
                
        logger.info("Evolution pipeline finished successfully.")