import json
import logging
import asyncio
import yaml
from pathlib import Path
from typing import Dict, List, Any
from pydantic import BaseModel, Field

from src.llm_platform.data_foundry.llm_client import LLMClient
from src.llm_platform.data_foundry.schemas import SFTPair, RawChunk
from pydantic import ValidationError
logger = logging.getLogger(__name__)


class LLMJudgeResult(BaseModel):
    """Schema for LLM-as-a-Judge evaluation result."""
    is_factual: bool = Field(
        description="True if the answer is factually correct strictly based on the source text, otherwise False."
    )
    reasoning: str = Field(
        description="Brief explanation of the verdict, pointing out specific errors if any."
    )


class DatasetEvaluator:
    """
    Evaluates dataset quality: calculates token/word statistics, 
    runs LLM-as-a-judge spot checks, and generates a markdown health report.
    """
    def __init__(
        self, 
        model_name: str,
        templates_dir: str | Path = "src/llm_platform/templates",
        max_concurrent_requests: int = 2
    ):
        self.client = LLMClient(model_name=model_name)
        self.semaphore = asyncio.Semaphore(max_concurrent_requests)
        
        self.templates_dir = Path(templates_dir)
        
        prompt_path = self.templates_dir / "prompts" / "judge_system.yaml"
        with open(prompt_path, "r", encoding="utf-8") as f:
            prompt_data = yaml.safe_load(f)
            self.system_prompt = prompt_data.get("system_prompt", "")

    def _calculate_basic_stats(self, dataset_file: Path) -> Dict[str, Any]:
        """Calculates length distributions and counts for the dataset."""
        lengths = []
        with open(dataset_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    pair = SFTPair.model_validate_json(line)
                    answer = next((m.content for m in pair.messages if m.role == "assistant"), "")
                    if answer:
                        lengths.append(len(answer.split()))
                except ValidationError:
                    continue

        if not lengths:
            return {"total_pairs": 0, "avg_words": 0, "max_words": 0, "min_words": 0}

        return {
            "total_pairs": len(lengths),
            "avg_words": round(sum(lengths) / len(lengths), 2),
            "max_words": max(lengths),
            "min_words": min(lengths)
        }

    async def _evaluate_single_pair(self, pair: SFTPair, source_text: str) -> bool:
        """Runs a single LLM-as-a-Judge check."""
        question = next((m.content for m in pair.messages if m.role == "user"), "")
        answer = next((m.content for m in pair.messages if m.role == "assistant"), "")
        user_prompt = (
            f"Source Text:\n{source_text}\n\n"
            f"User Question:\n{question}\n\n"
            f"Assistant Answer:\n{answer}\n\n"
            "Evaluate the answer."
        )

        async with self.semaphore:
            try:
                result = await self.client.generate_structured_data(
                    system_prompt=self.system_prompt,
                    user_prompt=user_prompt,
                    response_model=LLMJudgeResult
                )
                await asyncio.sleep(2)
                return result.is_factual
            except Exception as e:
                logger.error(f"Spot check failed: {e}")
                return False

    async def run_evaluation(
        self, 
        sanitized_sft_file: str | Path, 
        chunks_file: str | Path,
        report_output_file: str | Path = None,
        sanitization_stats: Dict[str, int] = None,
    ):
        """Generates the final dataset_health.md report."""
        if sanitization_stats is None:
            sanitization_stats = {}
            
        sft_path = Path(sanitized_sft_file)
        chunks_path = Path(chunks_file)

        # 1. Calculate basic statistics
        stats = self._calculate_basic_stats(sft_path)
        
        # 2. Load source chunks for spot-checking
        chunks_mapping = {}
        if chunks_path.exists():
            with open(chunks_path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        chunk = RawChunk.model_validate_json(line)
                        chunks_mapping[chunk.chunk_id] = chunk.text
                    except ValidationError:
                        continue

        # 3. Select 5% random samples for Spot Checking
        all_pairs = []
        with open(sft_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    all_pairs.append(SFTPair.model_validate_json(line))
                except ValidationError:
                    continue
        
        sample_size = max(1, int(len(all_pairs) * 0.05))
        import random
        sampled_pairs = random.sample(all_pairs, sample_size)
        
        logger.info(f"Running LLM-as-a-Judge on {sample_size} samples...")
        
        # 4. Run LLM evaluation concurrently
        tasks = []
        for pair in sampled_pairs:
            chunk_id = pair.source_chunk_id
            source_text = chunks_mapping.get(chunk_id, "No source text available.")
            tasks.append(self._evaluate_single_pair(pair, source_text))
            
        eval_results = await asyncio.gather(*tasks)
        factual_count = sum(eval_results)
        factual_rate = round((factual_count / len(eval_results)) * 100, 2) if eval_results else 0.0

        # 5. Generate Markdown Report
        status_icon = '✅ PASSED' if factual_rate >= 90.0 else '⚠️ WARNING (Accuracy < 90%)'
        rejection_rate = round(100 - (sanitization_stats.get('passed', 0) / max(1, sanitization_stats.get('total_processed', 1)) * 100), 2)

        report_template_path = self.templates_dir / "reports" / "dataset_health.md"
        with open(report_template_path, "r", encoding="utf-8") as f:
            template_content = f.read()
            
        report_content = template_content.format(
            total_processed=sanitization_stats.get('total_processed', 0),
            passed=sanitization_stats.get('passed', 0),
            rejection_rate=rejection_rate,
            rejected_structure=sanitization_stats.get('rejected_structure', 0),
            rejected_length=sanitization_stats.get('rejected_length', 0),
            rejected_refusal=sanitization_stats.get('rejected_refusal', 0),
            rejected_duplicate=sanitization_stats.get('rejected_duplicate', 0),
            total_pairs=stats.get('total_pairs', 0),
            avg_words=stats.get('avg_words', 0),
            max_words=stats.get('max_words', 0),
            min_words=stats.get('min_words', 0),
            sample_size=sample_size,
            factual_rate=factual_rate,
            status_icon=status_icon
        )

        if report_output_file:
            report_path = Path(report_output_file)
            with open(report_path, "w", encoding="utf-8") as f:
                f.write(report_content)
            logger.info(f"Dataset health report saved to {report_path}")
            
        return {
            "factual_rate": factual_rate,
            "report_content": report_content
        }
        