from dataclasses import dataclass, fields
from pprint import pformat
from typing import Any

import torch
from torch.utils.data import Dataset
from transformers.generation.utils import GenerateDecoderOnlyOutput
from transformers.tokenization_utils import BatchEncoding

from .typing import SEQUENCE, Tensor


@dataclass
class BaseInput:
    _id: str

    def to_json(self):
        return self.__dict__
    
    def __post_init__(self):
        if not isinstance(self._id, str):
            self._id = str(self._id)

    def __len__(self):
        raise NotImplementedError
    
    def get_prompt(self) -> str:
        raise NotImplementedError
    


@dataclass
class SimpleInput(BaseInput):
    prompt: str
    
    def __len__(self):
        return len(self.prompt)
    
    def get_prompt(self) -> str:
        return self.prompt
    

@dataclass
class SeqProbInput(BaseInput):
    prefix: str
    suffix: str
    eos_token_id: int
    length_penalty: float = 0.0

    def __len__(self):
        return len(self.prefix + self.suffix)
    
    # As function for readability
    def set_field__suffix_tokens(self, suffix_tokens: Tensor):
        setattr(self, "suffix_tokens", suffix_tokens)


class BaseDataset(Dataset):
    prompts: list[str]

    def __init__(self, inputs: list[BaseInput]):
        self.inputs = inputs

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx]


class BaseDataCollator:
    def __init__(
        self,
        tokenizer,
    ):
        self.tokenizer = tokenizer

    def __call__(self, inputs: list[BaseInput]) -> tuple[list[str], dict]:
        prompts = [_input.get_prompt() for _input in inputs]
        tokenizer_output = self.tokenizer(
            prompts, return_tensors="pt", padding="longest"
        )
        return inputs, tokenizer_output


@dataclass
class BaseInstance:
    def __init__(
        self, input_kwargs: dict, generation_kwargs, output_kwargs: dict[str] = None
    ):
        for _fields, prefix in [
            (input_kwargs, "input_kwargs"),
            (output_kwargs, "output_kwargs"),
            (generation_kwargs, "generation_kwargs"),
        ]:
            self.register_fields(_fields, prefix)

    @staticmethod
    def init_from_dict(instance_dict: dict) -> "BaseInstance":
        input_kwargs = {
            key.replace("input_kwargs__", ""): value
            for key, value in instance_dict.items()
            if key.startswith("input_kwargs__")
        }
        output_kwargs = {
            key.replace("output_kwargs__", ""): value
            for key, value in instance_dict.items()
            if key.startswith("output_kwargs__")
        }
        generation_kwargs = {
            key.replace("generation_kwargs__", ""): value
            for key, value in instance_dict.items()
            if key.startswith("generation_kwargs__")
        }
        return BaseInstance(
            input_kwargs=input_kwargs,
            generation_kwargs=generation_kwargs,
            output_kwargs=output_kwargs,
        )

    def register_fields(self, fields, prefix: str):
        if isinstance(fields, dict):
            for key, value in fields.items():
                setattr(self, f"{prefix}__{key}", value)
        elif fields is None:
            return
        else:
            raise ValueError

    def __repr__(self):
        return self.__class__.__name__ + "(\n" + pformat(self.__dict__, depth=2) + ")\n"

    def to_json(self):
        return self.__dict__

    def extract_category(self, prefix: str) -> dict:
        # Extract fields with the given prefix into a dictionary
        category_dict = {
            key.replace(f"{prefix}__", ""): value
            for key, value in self.__dict__.items()
            if key.startswith(f"{prefix}__")
        }
        return category_dict

    @staticmethod
    def dict_to_inputs(input_dict: dict, cls) -> "BaseInput":
        # Create a BaseInput instance from a dictionary, removing prefixes
        input_kwargs = {
            key.replace("input_kwargs__", ""): value
            for key, value in input_dict.items()
            if key.startswith("input_kwargs__")
        }
        return cls(**input_kwargs)


@dataclass
class BatchGenerationResult:
    inputs: list[BaseInput]
    model_inputs: BatchEncoding
    generation_kwargs: dict[str, Any]
    outputs: GenerateDecoderOnlyOutput

    def remove_input_instance_ids(self):
        for input_instance in self.inputs:
            input_instance.remove_instance_id()



class CollatorWithPositionIds(BaseDataCollator):

    def __call__(self, inputs: list[SeqProbInput]) -> tuple[list[str], dict]:
        prefix_tokenized = [
            self.tokenizer.encode(
                _input.prefix, add_special_tokens=True, return_tensors="pt"
            )
            for _input in inputs
        ]
        suffixes_tokenized = [
            self.tokenizer.encode(
                _input.suffix, add_special_tokens=False, return_tensors="pt"
            )
            for _input in inputs
        ]

        # concat prefix and suffixes
        tokenized = [
            prefix.squeeze(0).tolist() + suffix.squeeze(0).tolist()
            for prefix, suffix in zip(prefix_tokenized, suffixes_tokenized)
        ]

        # pad tokenized sequences to the longest sequence
        max_length = max(len(seq) for seq in tokenized)
        padded = [
            [self.tokenizer.pad_token_id] * (max_length - len(seq)) + seq
            for seq in tokenized
        ]
        padded = torch.tensor(padded, dtype=torch.long)


        # create attention mask
        attention_mask = [
            [0] * (max_length - len(seq)) + [1] * len(seq) for seq in tokenized
        ]
        attention_mask = torch.tensor(attention_mask, dtype=torch.long)

        position_ids = attention_mask.cumsum(axis=-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)

        tokenizer_output = {
            "input_ids": padded,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
        }

        for _input, tokenized_seq in zip(inputs, suffixes_tokenized):
            _input.set_field__suffix_tokens(tokenized_seq.squeeze(0))
        return inputs, tokenizer_output
