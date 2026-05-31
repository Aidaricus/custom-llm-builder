import asyncio
import logging
import uuid
from pathlib import Path
from typing import List, Optional

from src.llm_platform.data_foundry.document_processor import DocumentProcessor
from src.llm_platform.data_foundry.llm_client import LLMClient
from src.llm_platform.data_foundry.schemas import SFTPair, LLMGeneratedContent

logger = logging.getLogger(__name__)

class DatasetGenerator:
    """
    Orchestrates the end-to-end pipeline: reading PDFs, chunking, 
    and generating SFT pairs via LLM.
    """
    def __init__(
        self,
        raw_data_dir: str | Path,
        output_dir: str | Path,
        max_concurrent_requests: int = 5,
    ):
        self.raw_data_dir = Path(raw_data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.processor = DocumentProcessor()
        self.client = LLMClient()
        
        self.semaphore = asyncio.Semaphore(max_concurrent_requests)
    
    async def _process_single_chunk(self, chunk_text: str, system_prompt: str) -> Optional[SFTPair]:
        """Worker function to process one chunk through the LLM."""
        async with self.semaphore:
            try:
                llm_content = await self.client.generate_structured_data(
                    system_prompt=system_prompt,
                    user_prompt=chunk_text,
                    response_model=LLMGeneratedContent
                )
                chunk_id = f"chunk_{abs(hash(chunk_text)) % 1000000:06d}"
                pair_id = f"pair_{uuid.uuid4().hex[:8]}"

                pair = SFTPair(
                    pair_id=pair_id,
                    source_chunk_id=chunk_id,
                    messages=llm_content.messages,
                    is_evolved=False
                )

                return pair
            except Exception as e:
                logger.error(f"Failed to generate pair for chunk: {e}")
                return None
    
    async def generate_dataset(self, system_prompt: str, output_filename: str = "sft_dataset.jsonl"):
        """Main pipeline to process all PDFs and save the dataset as JSONL."""
        
        pdf_files = list(self.raw_data_dir.glob("*.pdf"))
        if not pdf_files:
            logger.warning(f"No PDF files found in {self.raw_data_dir}")
            return

        all_chunks = []
        
        # 1. Parse all documents
        for pdf_path in pdf_files:
            logger.info(f"Extracting chunks from: {pdf_path.name}")
            chunks = self.processor.process_file(pdf_path)
            all_chunks.extend(chunks)

        logger.info(f"Total chunks extracted: {len(all_chunks)}. Starting LLM generation...")

        # 2. Run LLM generation concurrently
        tasks = [
            self._process_single_chunk(chunk.text, system_prompt) 
            for chunk in all_chunks
        ]
        
        # asyncio.gather runs all tasks and waits for them to finish
        results = await asyncio.gather(*tasks)

        # Filter out any chunks that failed
        valid_pairs: List[SFTPair] = [res for res in results if res is not None]

        # 3. Save results to JSONL
        output_path = self.output_dir / output_filename
        logger.info(f"Writing dataset to {output_path}...")
        
        with open(output_path, "w", encoding="utf-8") as f:
            for pair in valid_pairs:
                # JSONL requires one JSON object per line
                f.write(pair.model_dump_json() + "\n")

        logger.info(f"Done! Successfully generated {len(valid_pairs)} SFT pairs.")