import asyncio
import json
import logging
import uuid
import yaml
from pathlib import Path
from typing import List, Optional

from src.llm_platform.data_foundry.llm_client import LLMClient
from src.llm_platform.data_foundry.schemas import SFTPair, LLMGeneratedContent, RawChunk

logger = logging.getLogger(__name__)

class DatasetGenerator:
    """
    Orchestrates the end-to-end pipeline: reading PDFs, chunking, 
    and generating SFT pairs via LLM.
    """
    def __init__(
        self,
        model_name: str,
        templates_dir: str | Path,
        max_concurrent_requests: int = 5,
    ):
        self.client = LLMClient(model_name=model_name)
        self.semaphore = asyncio.Semaphore(max_concurrent_requests)
        self.templates_dir = Path(templates_dir)
        
        # Загружаем промпт из конфигурации
        prompt_path = self.templates_dir / "prompts" / "generator_system.yaml"
        try:
            with open(prompt_path, "r", encoding="utf-8") as f:
                prompt_data = yaml.safe_load(f)
                self.system_prompt = prompt_data.get("system_prompt", "")
        except Exception as e:
            logger.error(f"Failed to load generator prompt from {prompt_path}: {e}")
            self.system_prompt = "Generate a QA pair based on the text." # Fallback
    
    async def _process_single_chunk(self, chunk: RawChunk) -> Optional[SFTPair]:
        """Worker function to process one chunk through the LLM."""
        async with self.semaphore:
            try:
                llm_content = await self.client.generate_structured_data(
                    system_prompt=self.system_prompt,
                    user_prompt=chunk.text,
                    response_model=LLMGeneratedContent
                )
                # chunk_id = f"chunk_{abs(hash(chunk.text)) % 1000000:06d}"
                pair_id = f"pair_{uuid.uuid4().hex[:8]}"

                pair = SFTPair(
                    pair_id=pair_id,
                    source_chunk_id=chunk.chunk_id,
                    messages=llm_content.messages,
                    is_evolved=False
                )

                return pair
            except Exception as e:
                logger.error(f"Failed to generate pair for chunk {chunk.chunk_id}: {e}")
                return None
    
    async def generate_dataset(self, chunks_file: str | Path, output_file: str | Path) -> List[SFTPair]:
        """Main pipeline to process all PDFs and save the dataset as JSONL."""
        chunks_path = Path(chunks_file)
        output_path = Path(output_file)
        
        if not chunks_path.exists():
            logger.error(f"Chunks artifact not found at: {chunks_path}")
            return []

        # 1. Читаем готовые чанки
        all_chunks = []
        with open(chunks_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    chunk_data = json.loads(line)
                    # Восстанавливаем объект RawChunk из JSON
                    chunk = RawChunk(**chunk_data)
                    all_chunks.append(chunk)
                except json.JSONDecodeError:
                    continue

        logger.info(f"Loaded {len(all_chunks)} chunks. Starting LLM generation...")

        # 2. Асинхронная генерация
        tasks = [self._process_single_chunk(chunk) for chunk in all_chunks]
        results = await asyncio.gather(*tasks)

        valid_pairs: List[SFTPair] = [res for res in results if res is not None]

        # 3. Сохранение артефакта
        logger.info(f"Writing dataset to {output_path}...")
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                for pair in valid_pairs:
                    f.write(pair.model_dump_json() + "\n")
            logger.info(f"Successfully generated and saved {len(valid_pairs)} SFT pairs.")
        except Exception as e:
            logger.error(f"Failed to save SFT dataset: {e}")

        return valid_pairs