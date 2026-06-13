import argparse
import asyncio
import logging
import sys
from pathlib import Path

# Импортируем наши классы
from src.llm_platform.data_foundry.document_processor import DocumentProcessor
from src.llm_platform.data_foundry.dataset_generator import DatasetGenerator
from src.llm_platform.data_foundry.evol_pipeline import EvolPipeline
from src.llm_platform.data_foundry.evaluator_data import DatasetEvaluator
from src.llm_platform.data_foundry.dataset_formatter import DatasetFormatter

# Настройка глобального логгера для консоли
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("DataFoundryCLI")

def parse_args():
    parser = argparse.ArgumentParser(
        description="LLM Data Foundry: End-to-end pipeline for generating and evaluating SFT datasets."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    parser.add_argument(
        "--input", 
        type=Path, 
        required=True, 
        help="Path to the input PDF document."
    )
    parser.add_argument(
        "--output-dir", 
        type=Path, 
        required=True,
        help="Directory where intermediate artifacts and final dataset will be saved."
    )
    parser.add_argument(
        "--model", 
        type=str, 
        default="google/gemma-4-31b-it:free", 
        help="Model name to use for generation and evaluation."
    )
    parser.add_argument(
        "--templates-dir", 
        type=Path, 
        default=Path("src/llm_platform/templates"), 
        help="Path to the directory containing YAML templates."
    )
    parser.add_argument(
        "--sanitize", 
        action="store_true", 
        help="Run DatasetSanitizer to filter out bad QA pairs before saving the final dataset."
    )
    parser.add_argument(
        "--steps", 
        type=str, 
        default="all", 
        help="Comma-separated steps to run: chunk,generate,evolve,evaluate. Default is 'chunk,generate,evaluate'."
    )
    parser.add_argument("--rag", action="store_true", help="Включить генерацию RAG датасета")

    fmt_parser = subparsers.add_parser("format", help="Format external raw datasets into target ChatML messages")
    fmt_parser.add_argument("--input", type=str, required=True, help="Path to raw dataset")
    fmt_parser.add_argument("--train-out", type=str, required=True, help="Path to save train dataset")
    fmt_parser.add_argument("--test-out", type=str, required=True, help="Path to save test dataset")
    fmt_parser.add_argument("--format", type=str, default="native", choices=["native", "alpaca", "sharegpt"], help="Input format type") 
    return parser.parse_args()

async def async_main(args):
    logger.info(f"Starting Data Foundry pipeline with model: {args.model}")
    logger.info(f"Target steps: {args.steps}")
    logger.info(f"Sanitize enabled: {args.sanitize}")
    
    # Создаем папку вывода, если ее нет
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Определение путей для артефактов
    chunks_file = args.output_dir / "1_chunks.jsonl"
    sft_base_file = args.output_dir / "2_sft_base.jsonl"
    sanitized_file = args.output_dir / "2_1_sft_sanitized.jsonl"
    evolved_file = args.output_dir / "3_sft_evolved.jsonl"
    report_file = args.output_dir / "4_dataset_report.md"
    
    steps = [s.strip().lower() for s in args.steps.split(",")]
    run_all = "all" in steps

    current_dataset = sft_base_file
    sanitization_stats = {}

    if run_all or "chunk" in steps:
        logger.info(">>> STEP 1: Document Chunking")
        processor = DocumentProcessor(chunk_size=500, chunk_overlap=100)
        processor.process_file(args.input, output_file=chunks_file)

    if run_all or "generate" in steps:
        logger.info(">>> STEP 2: SFT Generation")
        generator = DatasetGenerator(model_name=args.model, templates_dir=args.templates_dir, rag_mode=args.rag)
        await generator.generate_dataset(chunks_file=chunks_file, output_file=sft_base_file)
        current_dataset = sft_base_file # Обновляем указатель

    if args.sanitize:
        logger.info(">>> HOOK: Data Sanitization")
        from src.llm_platform.data_foundry.sanitizer import DatasetSanitizer
        sanitizer = DatasetSanitizer()
        sanitization_stats = sanitizer.sanitize_sft(input_file=current_dataset, output_file=sanitized_file)
        
        current_dataset = sanitized_file

    if run_all or "evolve" in steps:
        logger.info(">>> STEP 3: Data Evolution")
        prompts_file = args.templates_dir / "prompts" / "evol_prompts.yaml"
        evol_pipeline = EvolPipeline(prompts_file=prompts_file, model_name=args.model)
        
        await evol_pipeline.run_evolution(input_file=current_dataset, output_file=evolved_file)
        current_dataset = evolved_file # Обновляем указатель

    if run_all or "evaluate" in steps:
        logger.info(">>> STEP 4: Dataset Evaluation")
        evaluator = DatasetEvaluator(model_name=args.model, templates_dir=args.templates_dir)
        
        await evaluator.run_evaluation(
            sanitized_sft_file=current_dataset,
            chunks_file=chunks_file,
            report_output_file=report_file,
            sanitization_stats=sanitization_stats
        )

    logger.info(f"✅ Pipeline completed successfully. Output saved to {args.output_dir}")
def main():
    args = parse_args()
    try:
        if args.command == "generate":
            asyncio.run(async_main(args))
        elif args.command == "format":
            logger.info(f"Starting formatting: {args.input} ({args.format} -> messages)")
            formatter = DatasetFormatter()
            formatter.format_sft(
                input_file=args.input,
                train_file=args.train_out,
                test_file=args.test_out,
                input_format=args.format
            )
    except KeyboardInterrupt:
        logger.warning("Pipeline interrupted by user.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()