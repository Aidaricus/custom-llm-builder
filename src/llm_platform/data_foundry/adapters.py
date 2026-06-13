import logging
from typing import Dict, Any, Callable

logger = logging.getLogger(__name__)

def adapt_alpaca(row: Dict[str, Any]) -> Dict[str, Any]:
    """Конвертирует формат Alpaca в ChatML (messages)"""
    user_content = row.get("instruction", "")
    if row.get("input"):
        user_content += "\n" + row["input"]
        
    return {
        "messages": [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": row.get("output", "")}
        ]
    }

def adapt_sharegpt(row: Dict[str, Any]) -> Dict[str, Any]:
    """Конвертирует ShareGPT в ChatML"""
    role_map = {"human": "user", "gpt": "assistant", "system": "system"}
    messages = []
    
    for msg in row.get("conversations", []):
        messages.append({
            "role": role_map.get(msg.get("from", "human"), "user"),
            "content": msg.get("value", "")
        })
        
    return {"messages": messages}

def adapt_native(row: Dict[str, Any]) -> Dict[str, Any]:
    """Оставляет формат без изменений (для наших синтетических данных)"""
    return {"messages": row.get("messages", [])}

ADAPTER_REGISTRY: Dict[str, Callable[[Dict[str, Any]], Dict[str, Any]]] = {
    "alpaca": adapt_alpaca,
    "sharegpt": adapt_sharegpt,
    "native": adapt_native,
}

def get_adapter(format_name: str) -> Callable:
    """Возвращает нужную функцию-адаптер по имени"""
    format_name = format_name.lower()
    if format_name not in ADAPTER_REGISTRY:
        supported = ", ".join(ADAPTER_REGISTRY.keys())
        raise ValueError(f"Unknown format: {format_name}. Supported: {supported}")
    return ADAPTER_REGISTRY[format_name]