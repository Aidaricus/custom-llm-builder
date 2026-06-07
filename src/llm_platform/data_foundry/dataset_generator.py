import asyncio
import json
import logging
import uuid
from tqdm import tqdm
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
                    chunk = RawChunk(**chunk_data)
                    all_chunks.append(chunk)
                except json.JSONDecodeError:
                    continue
        
        processed_chunk_ids = set()
        if output_path.exists():
            with open(output_path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        # Используем Pydantic для надежного парсинга
                        pair = SFTPair.model_validate_json(line)
                        processed_chunk_ids.add(pair.source_chunk_id)
                    except Exception:
                        continue
            
            logger.info(f"Found existing dataset. Resuming... {len(processed_chunk_ids)} chunks already processed.")

        chunks_to_process = [c for c in all_chunks if c.chunk_id not in processed_chunk_ids]
        
        if not chunks_to_process:
            logger.info("All chunks have already been processed. Skipping generation.")
            return []

        logger.info(f"Loaded {len(all_chunks)} total chunks. Remaining to process: {len(chunks_to_process)}.")


        tasks = [self._process_single_chunk(chunk) for chunk in chunks_to_process]
        valid_pairs: List[SFTPair] = []

        try:
            # ВАЖНО: открываем в режиме "a" (дозапись), чтобы не стереть предыдущие 31 пару
            with open(output_path, "a", encoding="utf-8") as f:
                for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="SFT Generation"):
                    res = await coro
                    if res is not None:
                        valid_pairs.append(res)
                        f.write(res.model_dump_json() + "\n")
                        f.flush() 

            logger.info(f"Successfully generated and appended {len(valid_pairs)} new SFT pairs.")
        except Exception as e:
            logger.error(f"Pipeline interrupted or failed during generation: {e}")

        return valid_pairs