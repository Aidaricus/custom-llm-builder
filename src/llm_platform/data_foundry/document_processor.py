import logging
from pathlib import Path
from typing import List, Union

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
        if len(text) < self.chunk_size or not separators:
            return [text]
        
        separator = separators[0]
        splits = text.split(separator)

        if len(splits) == 1:
            return self._split_text(text, separators[1:])
        
        chunks = []
        current_chunk = ""

        for part in splits:
            part_with_sep = part + separator if part != splits[-1] else part

            if len(current_chunk) + len(part_with_sep) <= self.chunk_size:
                current_chunk += part_with_sep
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = part_with_sep

        if current_chunk:
            chunks.append(current_chunk.strip())

        final_chunks = []
        for chunk in chunks:
            if len(chunk) > self.chunk_size:
                final_chunks.extend(self._split_text(chunk, separators[1:]))
            else:
                final_chunks.append(chunk)

        return final_chunks
    
    def _apply_overlap(self, text_chunks: List[str]) -> List[str]:
        """Applies a sliding window overlap to adjacent chunks to prevent context loss."""
        if not text_chunks:
            return []
            
        overlapped_chunks = []
        for i in range(len(text_chunks)):
            if i == 0:
                overlapped_chunks.append(text_chunks[i])
                continue

            # Take the end of the previous chunk as the overlap prefix
            prev_chunk = text_chunks[i-1]
            if len(prev_chunk) > self.chunk_overlap:
                rough_overlap = prev_chunk[-self.chunk_overlap:]

                # Find the first space to avoid starting with a cut word (like "aded")
                first_space = rough_overlap.find(" ")
                if first_space != -1:
                    overlap_prefix = rough_overlap[first_space:].lstrip()
                else:
                    overlap_prefix = rough_overlap

            current_chunk = overlap_prefix + " " + text_chunks[i]
            # Ensure we don't massively exceed chunk_size due to overlap
            current_chunk = current_chunk.replace("\n", " ").replace("  ", " ")

            if len(current_chunk) > self.chunk_size + self.chunk_overlap:
                current_chunk = current_chunk[:self.chunk_size]
                # Also ensure we don't end on a cut word
                last_space = current_chunk.rfind(" ")
                if last_space != -1:
                    current_chunk = current_chunk[:last_space]
                
            overlapped_chunks.append(current_chunk.strip())
            
        return overlapped_chunks
    
    def _read_pdf(self, file_path: Path) -> str:
        """Extracts plain text from a PDF file."""
        text = ""
        try:
            with fitz.open(str(file_path)) as doc:
                for page in doc:
                    text += page.get_text("text") + "\n"
        except Exception as e:
            logger.error(f"Failed to read PDF {file_path}: {e}")
            raise
        return text
    
    def process_file(self, file_path: Union[str, Path]) -> List[RawChunk]:
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

        raw_text_chunks = self._split_text(full_text, self.separators)
        overlapped_texts = self._apply_overlap(raw_text_chunks)
        
        raw_chunks = []
        for i, txt in enumerate(overlapped_texts):
            if not txt.strip():
                continue
                
            chunk = RawChunk(
                text=txt,
                source_doc=path.name,
                metadata={
                    "chunk_index": i, 
                    "total_chars": len(txt)
                }
            )
            raw_chunks.append(chunk)

        return raw_chunks