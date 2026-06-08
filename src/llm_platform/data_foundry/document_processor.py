import logging
from pathlib import Path
from typing import List, Union
import hashlib
import fitz  # PyMuPDF
from .schemas import RawChunk

logger = logging.getLogger(__name__)

class DocumentProcessor:
    """
    Handles reading raw documents (PDF, TXT, MD) and splitting them 
    into semantic chunks using a recursive character splitting algorithm.
    """

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize the DocumentProcessor.
        
        Args:
            chunk_size: Maximum number of characters per chunk.
            chunk_overlap: Number of overlapping characters to preserve context 
                           between adjacent chunks.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        # Hierarchy of separators: Paragraphs -> Lines -> Sentences -> Words
        self.separators = ["\n\n", "\n", ". ", " "]

    def _split_text(self, text: str, separators: List[str]) -> List[str]:
        """Recursively splits text to maintain semantic boundaries."""
        if len(text) <= self.chunk_size or not separators:
            return [text]
            
        separator = separators[0]
        if separator not in text:
            return self._split_text(text, separators[1:])
            
        splits = text.split(separator)
        good_splits = []
        
        for part in splits:
            part_with_sep = part + separator if part != splits[-1] else part
            
            if len(part_with_sep) > self.chunk_size:
                if len(separators) > 1:
                    good_splits.extend(self._split_text(part_with_sep, separators[1:]))
                else:
                    good_splits.append(part_with_sep)
            else:
                if part_with_sep.strip():
                    good_splits.append(part_with_sep)
                
        return good_splits
    
    def _merge_splits_with_overlap(self, splits: List[str]) -> List[str]:
        """Combines semantic units into chunks, applying overlap via whole units."""
        chunks = []
        current_chunk = []
        current_length = 0
        
        for split in splits:
            split_len = len(split)
            
            if current_length + split_len > self.chunk_size and current_length > 0:
                chunks.append("".join(current_chunk).strip())
                
                overlap_length = 0
                overlap_chunk = []
                
                for prev_split in reversed(current_chunk):
                    if overlap_length + len(prev_split) <= self.chunk_overlap:
                        overlap_chunk.insert(0, prev_split)
                        overlap_length += len(prev_split)
                    else:
                        break
                
                current_chunk = overlap_chunk
                current_length = overlap_length
                
            current_chunk.append(split)
            current_length += split_len
            
        if current_chunk:
            chunks.append("".join(current_chunk).strip())
            
        return chunks
    
    def _read_pdf(self, file_path: Path) -> str:
        """Extracts plain text from a PDF file."""
        text = ""
        try:
            with fitz.open(str(file_path)) as doc:
                for page in doc:
                    blocks = page.get_text("blocks")
                    blocks.sort(key=lambda b: (b[1], b[0]))
                    for b in blocks:
                        block_text = b[4].strip()
                        if block_text:
                            text += block_text + "\n\n"
        except Exception as e:
            logger.error(f"Failed to read PDF {file_path}: {e}")
            raise
        return text
    
    def process_file(self, file_path: Union[str, Path], output_file: Union[str, Path, type(None)] = None) -> List[RawChunk]:
        """
        Reads a document, splits it, and returns standardized RawChunk objects.
        
        Args:
            file_path: Path to the target document.
            
        Returns:
            List of RawChunk objects ready for the SFT generation pipeline.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        full_text = ""
        if path.suffix.lower() == ".pdf":
            full_text = self._read_pdf(path)
        else:
            with open(path, "r", encoding="utf-8") as f:
                full_text = f.read()

        semantic_splits = self._split_text(full_text, self.separators)
        overlapped_texts = self._merge_splits_with_overlap(semantic_splits)
        
        raw_chunks = []
        for i, txt in enumerate(overlapped_texts):
            if not txt.strip():
                continue
            deterministic_id = hashlib.md5(txt.encode('utf-8')).hexdigest()[:16]
            chunk = RawChunk(
                chunk_id=f"chunk_{deterministic_id}",
                text=txt,
                source_doc=path.name,
                metadata={
                    "chunk_index": i, 
                    "total_chars": len(txt)
                }
            )
            raw_chunks.append(chunk)
        if output_file:
            output_path = Path(output_file)
            try:
                with open(output_path, "w", encoding="utf-8") as f:
                    for chunk in raw_chunks:
                        # RawChunk это Pydantic модель, используем model_dump_json()
                        f.write(chunk.model_dump_json() + "\n")
                logger.info(f"Successfully saved {len(raw_chunks)} chunks to {output_path}")
            except Exception as e:
                logger.error(f"Failed to write chunks to {output_path}: {e}")

        return raw_chunks