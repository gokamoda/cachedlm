from x_get_hidden_states import main, get_example_prompts
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from test.test_models import TEST_MODELS
import pytest
import torch

@pytest.mark.parametrize(
    "model_name_or_path",
    TEST_MODELS,
)
def test_chaching(model_name_or_path):
    model_name_str = model_name_or_path.replace("/", "--").replace(".", "-")
    cache_jsonl_path = Path(f"data/test_generation_cache_{model_name_str}.jsonl")

    # delete cache
    if cache_jsonl_path.exists():
        cache_jsonl_path.unlink()

    prompts = get_example_prompts()

    first_run_results = []
    for result in main(prompts=prompts, cache_jsonl_path=cache_jsonl_path, model_name_or_path=model_name_or_path):
        first_run_results += [result]

    second_run_results = []
    for result in main(prompts=prompts, cache_jsonl_path=cache_jsonl_path, model_name_or_path=model_name_or_path):
        second_run_results += [result]

    assert first_run_results == second_run_results

    
@pytest.mark.parametrize(
    "model_name_or_path",
    TEST_MODELS,
)
def test_hidden_state(model_name_or_path):
    model_name_str = model_name_or_path.replace("/", "--")
    cache_jsonl_path = Path(f"data/test_hiddenstate_cache_{model_name_str}.jsonl")
    ptcache_dir = Path(f'ptcache/{model_name_str}')

    prompts = get_example_prompts()[0:1]  # test only one prompt for simplicity

    token_position = -1
    layer_position = -2

    results = []
    for result in main(
        prompts=prompts,
        cache_jsonl_path=cache_jsonl_path,
        model_name_or_path=model_name_or_path,
        ptcache_dir=ptcache_dir
    ):
        results += result

    cached = torch.load(results[0].output_kwargs__hidden_state)

    # normal run
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token

    inputs = tokenizer(prompts, return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    hidden_states = outputs.hidden_states[layer_position][0, token_position, :]
    print(cached.shape)
    print(hidden_states.shape)

    assert torch.allclose(cached, hidden_states, atol=1e-4, rtol=1e-4)


    
    


    
