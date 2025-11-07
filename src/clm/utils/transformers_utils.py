from transformers import AutoTokenizer


def is_chat_model(tokenizer: AutoTokenizer) -> bool:
    return tokenizer.chat_template is not None
