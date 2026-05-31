import json
import logging
from typing import Type, TypeVar

import openai
from openai import AsyncOpenAI
from pydantic import BaseModel, ValidationError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential
)

from src.llm_platform.config import OPENROUTER_API_KEY

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class LLMClient:
    def __init__(self, model_name: str = "openai/gpt-oss-120b"):
        if not OPENROUTER_API_KEY:
            raise ValueError(
                "OPENROUTER_API_KEY is not set. Please check your .env file."
            )
        self.model_name = model_name

        self.client = AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=OPENROUTER_API_KEY,
        )

        logger.info(f"LLMClient initialized with model: {self.model_name}")

    def _log_retry(retry_state):
        logger.warning(
            f"Network or API error. Retrying LLM request... "
            f"(Attempt {retry_state.attempt_number})"
        )

    @retry(
        # Wait 4 seconds, then 8, then 16, up to 60 seconds max
        wait=wait_exponential(multiplier=1, min=4, max=60),
        # Stop trying after 5 attempts
        stop=stop_after_attempt(5),
        # Only retry on specific transient errors
        retry=retry_if_exception_type(
            (
                openai.RateLimitError,
                openai.APIConnectionError,
                openai.InternalServerError,
            )
        ),
        before_sleep=_log_retry,
    )
    async def generate_structured_data(
        self, system_prompt: str, user_prompt: str, response_model: Type[T]
    ) -> T:
        """
        Sends a prompt to the LLM and guarantees the output matches the Pydantic schema.
        """
        schema_json = response_model.model_json_schema()
        full_system_prompt = (
            f"{system_prompt}\n\n"
            f"You must respond ONLY with a valid JSON object that strictly adheres "
            f"to the following JSON schema:\n{json.dumps(schema_json)}"
        )
        try:
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": full_system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                # Force JSON output mode (supported by most OpenRouter models)
                response_format={"type": "json_object"},
                temperature=0.3, # Low temperature for more deterministic/factual output
            )

            raw_content = response.choices[0].message.content
            
            if not raw_content:
                raise ValueError("Received empty response from the model.")

            # Validate the JSON string against our Pydantic model
            parsed_data = response_model.model_validate_json(raw_content)
            return parsed_data

        except ValidationError as e:
            logger.error(f"Failed to parse LLM response into {response_model.__name__}: {e}")
            logger.debug(f"Raw response was: {raw_content}")
            raise