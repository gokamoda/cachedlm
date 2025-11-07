from pathlib import Path
from time import sleep

import torch
from transformers import AutoTokenizer

from .data import (
    BaseInputs,
    BaseInstance,
    BatchGenerationResult,
    SeqProbInputs,
)
from .typing import BATCH, HIDDEN_DIM, SEQUENCE, VOCAB, Tensor


def postprocess_simple_generation(
    tokenizer: AutoTokenizer,
    batch_result: BatchGenerationResult,
) -> list[BaseInstance]:
    decoded = tokenizer.batch_decode(
        batch_result.outputs.sequences, skip_special_tokens=False
    )

    instances = [
        BaseInstance(
            input_kwargs=batch_result.inputs[i].__dict__,
            generation_kwargs=batch_result.generation_kwargs,
            output_kwargs={
                "generated_sequence": decoded[i].split(batch_result.inputs[i].prompt)[
                    -1
                ]
            },
        )
        for i in range(len(batch_result.inputs))
    ]
    return instances


def postprocess_get_label_probs(
    tokenizer: AutoTokenizer,
    labels: list[str],
    batch_result: BatchGenerationResult,
) -> list[BaseInstance]:
    logits: Tensor[BATCH, VOCAB] = batch_result.outputs.logits[0]
    labels_ids = [
        tokenizer.encode(label, add_special_tokens=False)[0] for label in labels
    ]
    logits = logits[:, labels_ids]
    probs = logits.softmax(dim=-1)

    instances = [
        BaseInstance(
            input_kwargs=batch_result.inputs[i].__dict__,
            generation_kwargs=batch_result.generation_kwargs,
            output_kwargs={"probabilities": probs[i].tolist()},
        )
        for i in range(len(batch_result.inputs))
    ]

    return instances


def postprocess_prob_generation(
    tokenizer,
    batch_result: BatchGenerationResult,
) -> list[BaseInstance]:
    max_suffix_length = max(len(_input.suffix_tokens) for _input in batch_result.inputs)

    # probabilities of next token prediction
    # SEQUENCE should be max_suffix_length + 1 (+1 for eos_token_id)
    probs: Tensor[BATCH, SEQUENCE, VOCAB] = torch.nn.functional.softmax(
        batch_result.outputs["logits"][:, -(max_suffix_length + 1) :, :], dim=-1
    )  # max_suffix_length+1 because we pad the suffixes with eos_token_id

    # positions to retrieve from probs
    next_token_ids: Tensor[BATCH, SEQUENCE] = torch.stack(
        [
            torch.nn.functional.pad(
                torch.nn.functional.pad(
                    _input.suffix_tokens,
                    (0, 1),
                    value=_input.eos_token_id,
                ),
                (max_suffix_length - len(_input.suffix_tokens), 0),
                value=-1,  # padding with -1 to avoid confusion with eos_token_id
            )
            for _input in batch_result.inputs
        ]
    )

    # mask for padding
    mask: Tensor[BATCH, SEQUENCE] = next_token_ids != -1
    mask = mask.to(probs.device)
    mask_noeos = mask.clone()
    mask_noeos[:, -1] = 0

    # replace padding with 0
    next_token_ids[next_token_ids == -1] = 0  # can be anything if positive

    # get probabilities of next token
    probs: Tensor[BATCH, SEQUENCE] = torch.gather(
        probs, -1, next_token_ids.to(probs.device).unsqueeze(-1)
    ).squeeze()

    sequence_probs = torch.prod(probs * mask + ~mask, dim=-1).tolist()
    sequence_probs_no_eos = torch.prod(
        probs * mask_noeos + ~mask_noeos, dim=-1
    ).tolist()

    for i in range(len(batch_result.inputs)):
        if batch_result.inputs[i].length_penalty is not None:
            sequence_probs[i] /= (
                mask[i].sum().item() ** batch_result.inputs[i].length_penalty
            )
            sequence_probs_no_eos[i] /= (
                mask_noeos[i].sum().item() ** batch_result.inputs[i].length_penalty
            )

    for i in range(len(batch_result.inputs)):
        del batch_result.inputs[i].suffix_tokens

    instances = [
        BaseInstance(
            input_kwargs=batch_result.inputs[i].__dict__,
            generation_kwargs=batch_result.generation_kwargs,
            output_kwargs={
                "probs": sequence_probs[i],
                "probs_no_eos": sequence_probs_no_eos[i],
            },
        )
        for i in range(len(batch_result.inputs))
    ]
    return instances


def postprocess_get_hidden_states(
    tokenizer: AutoTokenizer,
    token_position: int,
    layer_position: int,
    tensor_cache_dir: Path,
    prompt_to_filename_converter: callable,
    batch_result: BatchGenerationResult,
) -> list[BaseInstance]:
    # hidden_states: TensorType[LAYER_PLUS_1, BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE] = torch.stack(batch_result.outputs.hidden_states[0])
    hidden_states: Tensor[BATCH, HIDDEN_DIM] = batch_result.outputs.hidden_states[0][
        layer_position
    ][:, token_position, :]

    # Normalize by l2 norm
    # hidden_states = torch.nn.functional.normalize(hidden_states, dim=-1)

    save_paths = [
        tensor_cache_dir.joinpath(prompt_to_filename_converter(_input.prompt))
        for _input in batch_result.inputs
    ]

    for hidden_state, save_path in zip(hidden_states, save_paths):
        torch.save(hidden_state.detach().cpu().clone(), save_path)
        sleep(0.1)

    instances = [
        BaseInstance(
            input_kwargs=_input.__dict__,
            generation_kwargs=batch_result.generation_kwargs,
            output_kwargs={"hidden_state": save_path.__str__()},
        )
        for _input, save_path, hidden_state in zip(
            batch_result.inputs, save_paths, hidden_states
        )
    ]

    return instances
