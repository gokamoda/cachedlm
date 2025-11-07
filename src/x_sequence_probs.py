from functools import partial
from typing import Generator

from cachedlm import (
    ModelCallWithCache,
    postprocess_prob_generation
)
from cachedlm.data import (
    CollatorWithPositionIds,
    BaseInputs,
    BaseInstance,
    SeqProbInputs
)


def get_example_data():
    data = [
        ("The capital of the United States of America is", " Washington D.C."),
        ("The capital of Germany is", " Berlin."),
        ("The capital of Italy is", " Rome."),
        ("The capital of the United States of America is", " Berlin."),
        ("The capital of Germany is", " Rome."),
        ("The capital of Italy is", " Washington D.C."),
        ("The capital of the United States of America is", " Rome."),
        ("The capital of Germany is", " Washington D.C."),
        ("The capital of Italy is", " Berlin."),
    ]

    return data


def main(
    cache_jsonl_path="data/classification_cache.jsonl",
    model_name_or_path="meta-llama/Llama-3.2-3B-Instruct",
) -> Generator[list[BaseInstance], None, None]:
    model = ModelCallWithCache(
        model_name_or_path=model_name_or_path,
        cache_jsonl_path=cache_jsonl_path,
    )

    data = get_example_data()
    prefixes, suffixes = zip(*data)


    result_generator = model.run_inference(
        cache_file_name="test-model-with-cache",
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
        batch_size=2,
        post_process_fn=partial(postprocess_prob_generation, model.tokenizer),
    )

    yield from result_generator


if __name__ == "__main__":
    for result in main():
        print(result)
