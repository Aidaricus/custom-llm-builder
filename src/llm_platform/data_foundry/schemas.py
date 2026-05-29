import uuid
from typing import Any, Dict, Optional, Literal, List
from pydantic import BaseModel, Field


class Message(BaseModel):
    """Base message in ChatML format."""
    role: Literal["system", "user", "assistant"] = Field(
        ..., 
        description="Role of the message sender."
    )
    content: str = Field(
        ..., 
        description="The actual text content of the message."
    )

class RawChunk(BaseModel):
    """Base model for a raw text chunk extracted from a source document."""
    
    chunk_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()), 
        description="Unique identifier for the chunk."
    )
    text: str = Field(
        ..., 
        description="The actual text content of the chunk."
    )
    source_doc: str = Field(
        ..., 
        description="Name of the source document (e.g., 'guide.md')."
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, 
        description="Additional metadata (e.g., position in text, tags)."
    )

class SFTPair(BaseModel):
    """Generated instruction pair (dialogue) for Supervised Fine-Tuning (SFT)."""
    
    pair_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()), 
        description="Unique identifier for the SFT pair."
    )
    source_chunk_id: str = Field(
        ..., 
        description="ID of the source RawChunk used to generate the prompt."
    )
    messages: List[Message] = Field(
        ..., 
        description="List of messages (typically system, user, assistant)."
    )
    is_evolved: bool = Field(
        default=False, 
        description="Flag indicating whether this prompt was generated via Evol-Instruct."
    )

class DPOTriplet(BaseModel):
    """Preference triplet for Direct Preference Optimization (DPO)."""
    
    triplet_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()), 
        description="Unique identifier for the DPO triplet."
    )
    source_pair_id: str = Field(
        ..., 
        description="ID of the source SFTPair containing the prompt and chosen response."
    )
    prompt: List[Message] = Field(
        ..., 
        description="Dialogue context (system and user messages)."
    )
    chosen: List[Message] = Field(
        ..., 
        description="The ideal or preferred response (assistant message)."
    )
    rejected: List[Message] = Field(
        ..., 
        description="The intentionally poor, hallucinated, or refused response (assistant message)."
    )
    reject_reason: Optional[str] = Field(
        default=None, 
        description="Reason for rejection (e.g., 'refusal', 'hallucination', 'too_short')."
    )
