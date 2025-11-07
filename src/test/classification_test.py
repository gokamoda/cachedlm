from x_classification import main
from pathlib import Path
import pytest
from test.test_models import TEST_MODELS

@pytest.mark.parametrize(
    "model_name_or_path",
    TEST_MODELS,
)
def test_chaching(model_name_or_path):
    model_name_str = model_name_or_path.replace("/", "--").replace(".", "-")
    cache_jsonl_path = Path(f"data/test_classification_cache_{model_name_str}.jsonl")

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

    


