import asyncio
import logging
import uuid
from pathlib import Path
from typing import List, Optional
import yaml

from src.llm_platform.data_foundry.llm_client import LLMClient
from src.llm_platform.data_foundry.schemas import SFTPair, DPOTriplet, LLMRejectedResponse, Message

logger = logging.getLogger(__name__)

class DPOPipeline:
    """
    Pipeline for generating DPO triplets by creating synthetic 'rejected' responses
    based on high-quality SFT pairs.
    """
    def __init__(
        self,
        input_file: str | Path,
        output_file: str | Path,
        prompts_file: str | Path,
        model_name: str,
        max_concurrent_requests: int = 1,
    ):
        self.input_file = Path(input_file)
        self.output_file = Path(output_file)
        self.prompts_file = Path(prompts_file)
        
        self.client = LLMClient(model_name=model_name)
        self.semaphore = asyncio.Semaphore(max_concurrent_requests)
        
        self.system_prompt = self._load_prompt()

    def _load_prompt(self) -> str:
        """Loads the DPO generation system prompt from the YAML configuration."""
        try:
            with open(self.prompts_file, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
                return data.get("dpo_prompts", {}).get("generate_rejected", "")
        except Exception as e:
            logger.error(f"Failed to load DPO prompt from {self.prompts_file}: {e}")
            return ""

    async def _process_single_pair(self, pair: SFTPair) -> Optional[DPOTriplet]:
        """Generates a rejected response for a single SFT pair and constructs a triplet."""
        if not self.system_prompt:
            logger.error("No DPO system prompt loaded.")
            return None

        prompt = next((m.content for m in pair.messages if m.role == "user"), "")
        chosen = next((m.content for m in pair.messages if m.role == "assistant"), "")
        
        if not prompt or not chosen:
            return None

        user_prompt = f"User Question:\n{prompt}\n\nIdeal Response (Chosen):\n{chosen}\n\nGenerate the rejected response."

        async with self.semaphore:
            try:
                llm_response = await self.client.generate_structured_data(
                    system_prompt=self.system_prompt,
                    user_prompt=user_prompt,
                    response_model=LLMRejectedResponse
                )
                
                await asyncio.sleep(5)
                
                triplet = DPOTriplet(
                    triplet_id=f"dpo_{uuid.uuid4().hex[:8]}",
                    source_pair_id=pair.pair_id,
                    prompt=[Message(role="user", content=prompt)],
                    chosen=[Message(role="assistant", content=chosen)],
                    rejected=[Message(role="assistant", content=llm_response.rejected_text)],
                    reject_reason=llm_response.reject_reason
                
                logger.debug(f"Generated DPO triplet for source pair {pair.pair_id}")
                return triplet
                
            except Exception as e:
                logger.error(f"Failed to generate DPO triplet for pair {pair.pair_id}: {e}")
                return None

    async def run_generation(self):
        """Reads SFT pairs, generates rejections, and saves the resulting DPO triplets."""
        if not self.input_file.exists():
            logger.error(f"Input file not found: {self.input_file}")
            return

        # 1. Read input data
        base_pairs: List[SFTPair] = []
        with open(self.input_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    pair = SFTPair.model_validate_json(line)
                    base_pairs.append(pair)
                except Exception as e:
                    logger.warning(f"Skipping invalid line: {e}")

        logger.info(f"Loaded {len(base_pairs)} pairs for DPO generation.")

        # 2. Process concurrently (constrained by semaphore)
        tasks = [self._process_single_pair(pair) for pair in base_pairs]
        results = await asyncio.gather(*tasks)

        valid_triplets: List[DPOTriplet] = [res for res in results if res is not None]

        # 3. Save to output
        logger.info(f"Writing {len(valid_triplets)} DPO triplets to {self.output_file}...")
        with open(self.output_file, "w", encoding="utf-8") as f:
            for triplet in valid_triplets:
                f.write(triplet.model_dump_json() + "\n")
                
        logger.info("DPO pipeline finished successfully.")