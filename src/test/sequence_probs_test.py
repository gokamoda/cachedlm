import pytest
from x_sequence_probs import main
from pathlib import Path
from test.test_models import TEST_MODELS
from transformers import AutoTokenizer
from cachedlm.data import CollatorWithPositionIds, BaseInputs, BaseDataset, SeqProbInputs
from torch.utils.data import DataLoader
from cachedlm import ModelCallWithCache
from cachedlm.postprocess import postprocess_prob_generation
from functools import partial
import torch

@pytest.mark.parametrize(
    "model_name_or_path",
    TEST_MODELS,
)
def test_chaching(model_name_or_path):
    model_name_str = model_name_or_path.replace("/", "--").replace(".", "-")
    cache_jsonl_path = Path(f"data/test_seqprob_cache_{model_name_str}.jsonl")

    # delete cache
    if cache_jsonl_path.exists():
        cache_jsonl_path.unlink()

    first_run_results = []
    for result in main(cache_jsonl_path=cache_jsonl_path, model_name_or_path=model_name_or_path):
        first_run_results += [result]

    second_run_results = []
    for result in main(cache_jsonl_path=cache_jsonl_path, model_name_or_path=model_name_or_path):
        second_run_results += [result]

    assert first_run_results == second_run_results

    
@pytest.mark.parametrize(
    "model_name_or_path",
    TEST_MODELS,
)
def test_collator(model_name_or_path):
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token

    prefixes = [
        "Hello, how are you",
        "The quick brown fox jumps over the lazy",
    ]
    suffixes = [
        "?",
        " dog.",
    ]
    batch_size = 2

    inputs = [
        SeqProbInputs(
            prefix=prefix,
            suffix=suffix,
            eos_token_id=tokenizer.eos_token_id,
            length_penalty=None,
        )
        for prefix, suffix in zip(prefixes, suffixes)
    ]

    dataset = BaseDataset(
        sorted(
            inputs,
            key=len,
            reverse=True,
        )
    )

    data_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        collate_fn=CollatorWithPositionIds(tokenizer=tokenizer),
        shuffle=False,
    )

    _inputs, _model_inputs= data_loader.__iter__().__next__()
    
    print(_model_inputs)

    assert _model_inputs["input_ids"][1,0] == tokenizer.pad_token_id, "Padding not applied correctly"
    assert _model_inputs["attention_mask"][1,0] == 0, "Attention mask not applied correctly"

    
    masked_positions = _model_inputs["attention_mask"] * -1 + 1
    assert _model_inputs['position_ids'][1].sum() == masked_positions[1].sum() + sum(list(range(_model_inputs['attention_mask'][1].sum()))) , "Position IDs not applied correctly"

def get_example_data():
    data = [
        ("The capital of the United States of America is", " Washington D.C."),
        ("The capital of Germany is", " Berlin."),
        ("The capital of Italy is", " Rome."),  # Test target
    ]

    return data


@pytest.mark.parametrize(
    "model_name_or_path",
    TEST_MODELS,
)
def test_batch(
    model_name_or_path
):
        
    model_name_str = model_name_or_path.replace("/", "--").replace(".", "-")
    cache_jsonl_path = Path(f"data/test_seqprob_cache_{model_name_str}.jsonl")

    # delete cache
    if cache_jsonl_path.exists():
        cache_jsonl_path.unlink()

    model = ModelCallWithCache(
        model_name_or_path=model_name_or_path,
        cache_jsonl_path=cache_jsonl_path,
    )

    data = get_example_data()


    # Run batch with longer sequences
    prefixes, suffixes = zip(*data)
    result_generator = model.run_inference(
        generation_kwargs={
            "return_dict_in_generate": True,
            "do_sample": False,
            "pad_token_id": model.tokenizer.eos_token_id,
            "max_new_tokens": 6,
            "output_logits": True,
        },
        inputs=[
            SeqProbInputs(
                prefix=prefix,
                suffix=suffix,
                eos_token_id=model.tokenizer.eos_token_id,
                length_penalty=None,
            )
            for prefix, suffix in zip(prefixes, suffixes)
        ],
        collator=CollatorWithPositionIds(model.tokenizer),
        batch_size=4,
        post_process_fn=partial(postprocess_prob_generation, model.tokenizer),
    )

    result = next(result_generator)
    prob1 = result[-1].output_kwargs__probs
    print(prob1)

    # Run without other sequences
    # delete cache
    if cache_jsonl_path.exists():
        cache_jsonl_path.unlink()
    data = data[-1:]  # Only keep the last one
    prefixes, suffixes = zip(*data)
    result_generator = model.run_inference(
        generation_kwargs={
            "return_dict_in_generate": True,
            "do_sample": False,
            "pad_token_id": model.tokenizer.eos_token_id,
            "max_new_tokens": 6,
            "output_logits": True,
        },
        inputs=[
            SeqProbInputs(
                prefix=prefix,
                suffix=suffix,
                eos_token_id=model.tokenizer.eos_token_id,
                length_penalty=None,
            )
            for prefix, suffix in zip(prefixes, suffixes)
        ],
        collator=CollatorWithPositionIds(model.tokenizer),
        batch_size=4,
        post_process_fn=partial(postprocess_prob_generation, model.tokenizer),
    )

    result = next(result_generator)
    prob2 = result[-1].output_kwargs__probs
    print(prob2)

    assert torch.isclose(torch.tensor(prob1), torch.tensor(prob2)), "Probabilities do not match!"