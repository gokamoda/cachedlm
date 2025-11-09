from functools import partial
from typing import Generator

from cachedlm import (
    JobManager,
    DeterministicModelWithCache,
    postprocess_simple_generation,
)
from cachedlm.data import (
    BaseDataCollator,
    SimpleInput,
    BaseInstance,
)


def get_example_prompt_template():
    return "{instruction}\n\n{demos}Text: {target}\nSentiment: "


def get_example_instruction():
    instruction = (
        "Predict the sentiment of following text in 0 and 1.\n"
        "Respond only with the integer without providing explanations or reasons for the prediction.\n\n"
        "Here are the definitions of the sentiments:\n"
        'Sentiment 0 means "Negative"\n'
        'Sentiment 1 means "Positive"'
    )
    return instruction


def get_example_targets():
    targets = [
        "This is the worst experience I have ever had.",
        "This is the worst experience I have ever had.",
        "This is the worst experience I have ever had.",
        "The service was okay, nothing special.",
        "I am extremely satisfied with the quality.",
        "I would not recommend this to anyone.",
        "The food was delicious and the ambiance was great.",
    ]
    return targets


def get_example_prompts():
    prompt_template = get_example_prompt_template()
    instruction = get_example_instruction()
    targets = get_example_targets()

    prompts = [
        prompt_template.format(instruction=instruction, demos="", target=target)
        for target in targets
    ]
    return prompts


def main(
    cache_jsonl_path="data/generation_cache.jsonl",
    model_name_or_path="meta-llama/Llama-3.2-3B-Instruct",
) -> Generator[list[BaseInstance], None, None]:
    model = DeterministicModelWithCache(
        model_name_or_path=model_name_or_path,
        cache_jsonl_path=cache_jsonl_path,
    )

    prompts = get_example_prompts()
    inputs=[SimpleInput(prompt=prompt, _id=_id) for _id, prompt in enumerate(prompts)]

    job_manager = JobManager(model=model)

    result_generator = job_manager.submit(
        generation_kwargs={
            "return_dict_in_generate": True,
            "do_sample": False,
            "pad_token_id": model.tokenizer.eos_token_id,
            "max_new_tokens": 6,
        },
        inputs=inputs,
        collator=BaseDataCollator(model.tokenizer),
        batch_size=2,
        post_process_fn=partial(postprocess_simple_generation, model.tokenizer),
    )

    yield from result_generator


if __name__ == "__main__":
    for result in main():
        print(result)
