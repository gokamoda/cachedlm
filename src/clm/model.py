import inspect
import os
from functools import partial
from pathlib import Path
from typing import Generator

import polars as pl
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm as std_tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from .data import (
    BaseDataCollator,
    BaseDataset,
    BaseInputs,
    BaseInstance,
    BatchGenerationResult,
)
from .typing import BATCH, Tensor
from .utils.int_utils import check_power_of_2
from .utils.logger import init_logging

tqdm = partial(std_tqdm, dynamic_ncols=True)

logger = init_logging(__name__)


class CacheReader:
    def __init__(
        self,
        cache_jsonl_path: str | Path,
    ):
        # convert to Path
        if isinstance(cache_jsonl_path, str):
            cache_jsonl_path = Path(cache_jsonl_path)
        self.cache_jsonl_path = cache_jsonl_path

    def get_inputs_to_compute(
        self,
        inputs: list[BaseInputs],
        generation_kwargs: dict,
    ):
        # format queries
        instances = [
            BaseInstance(_input.__dict__, generation_kwargs) for _input in inputs
        ]
        df_queries = pl.DataFrame([instance.to_json() for instance in instances])
        input_columns = df_queries.columns.copy()
        print(f"Input columns: {input_columns}")
        df_queries = df_queries.select(input_columns).unique(subset=input_columns)

        # prepare results dataframe
        precomputed_results = pl.DataFrame()

        if self.cache_jsonl_path.exists():  # load cache
            if df_queries.height != df_queries.unique(subset=input_columns).height:
                print(
                    f"duplicate queries found: {df_queries.height - df_queries.unique(subset=input_columns).height}",
                    df_queries.with_columns(
                        pl.count().over(input_columns).alias("count")
                    )
                    .filter(pl.col("count") > 1)[0]
                    .to_dicts(),
                )

            logger.info(f"Loading precomputed results from {self.cache_jsonl_path}")
            precomputed_results = pl.read_ndjson(self.cache_jsonl_path)

            logger.info(f"{df_queries.height=}")
            precomputed_results = df_queries.join(
                precomputed_results,
                on=input_columns,
                how="inner",
            )
            logger.info(f"{precomputed_results.height=}")

            df_queries = df_queries.join(
                precomputed_results, on=input_columns, how="left"
            )

            df_queries = (
                df_queries.filter(
                    pl.any_horizontal(pl.all().is_null())
                )  # get rows with null values in any of the columns
                .select(input_columns)  # select only input columns
                .unique(subset=input_columns)  # avoid duplicates
            )

        else:
            logger.info(f"Cache file {self.cache_jsonl_path} does not exist.")

        self.precomputed_df = precomputed_results

        inputs_to_compute = [
            BaseInstance.dict_to_inputs(row, cls=inputs[0].__class__) for row in df_queries.to_dicts()
        ]
        return inputs_to_compute

    def get_batch_results(
        self, batch_size: int
    ) -> Generator[list[BaseInstance], None, None]:
        if self.precomputed_df.height == 0:
            return

        results = [
            BaseInstance.init_from_dict(row) for row in self.precomputed_df.to_dicts()
        ]

        for i in range(0, len(results), batch_size):
            yield results[i : i + batch_size]

    def cache_results(self, results: list[BaseInstance]):
        new_results = [instance.to_json() for instance in results]

        # Create and write
        if not self.cache_jsonl_path.exists():
            logger.info(
                f"Cache file {self.cache_jsonl_path} does not exist. Creating new one."
            )
            new_results_df = pl.DataFrame(new_results)
            new_results_df.write_ndjson(self.cache_jsonl_path)
            logger.info(f"Saved new results to {self.cache_jsonl_path}")
            return new_results_df

        # Update
        try:
            pre_computed_results = pl.read_ndjson(self.cache_jsonl_path)
        except Exception as e:
            logger.error(e)
            new_results_df = pl.DataFrame(new_results)
            new_results_df.write_ndjson(self.cache_jsonl_path)
            return new_results_df

        new_results_df = pl.DataFrame(new_results)
        pl.concat([pre_computed_results, new_results_df], how="vertical").write_ndjson(
            self.cache_jsonl_path
        )


class DeterministicModelWithCache:
    generation_kwargs = {
        "do_sample": False,
        "num_beams": 1,
        "num_return_sequences": 1,
        "return_dict_in_generate": True,
        "output_logits": True,
    }

    def __init__(
        self,
        model_name_or_path: str,
        cache_jsonl_path: str | Path,
        model: AutoModelForCausalLM = None,
        tokenizer: AutoTokenizer = None,
    ):
        self._init_cache_path(cache_jsonl_path)
        self._init_model(model_name_or_path, model)
        self._init_tokenizer(model_name_or_path, tokenizer)

    def _init_cache_path(self, cache_jsonl_path: str | Path) -> Path:
        if isinstance(cache_jsonl_path, str):
            cache_jsonl_path = Path(cache_jsonl_path)

        cache_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_jsonl_path = cache_jsonl_path

    def _init_model(self, model_name_or_path: str, model: AutoModelForCausalLM = None):
        self.model = model  # load when needed
        self.model_name_or_path = model_name_or_path

    def _init_tokenizer(self, model_name_or_path: str, tokenizer: AutoTokenizer = None):
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(
                model_name_or_path, padding_side="left"
            )
            tokenizer.pad_token_id = tokenizer.eos_token_id
        self.tokenizer = tokenizer

    def load_model(self, tensor_type=None):
        if torch.cuda.is_available():
            if torch.cuda.device_count() > 1:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name_or_path, device_map="auto"
                )
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name_or_path
                ).to("cuda")
        else:
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name_or_path)

        self.model.eval()

    def __repr__(self):
        return f"DeterministicModelWithCache(model_name_or_path={self.model_name_or_path}, cache_jsonl_path={self.cache_jsonl_path})"

    def set_instance_ids(self, inputs: list[BaseInputs]):
        for i, _input in enumerate(inputs):
            _input.instance_id = i

        return inputs

    def model_generate(self, model_input_kwargs, generation_kwargs):
        model_input_kwargs = model_input_kwargs.to(self.model.device)
        return self.model.generate(**model_input_kwargs, **generation_kwargs)

    def batch_generate(
        self,
        inputs: list[BaseInputs],
        generation_kwargs: dict,
        collator: BaseDataCollator,
        batch_size: int = 32,
        sort_by_length: bool = False,
    ) -> Generator[BatchGenerationResult, None, None]:
        frame = inspect.currentframe()
        outer_frames = inspect.getouterframes(frame)

        for outer in outer_frames:
            filename = outer.filename
            filename_short = os.path.relpath(filename, os.getcwd())
            if not any(
                skip in filename
                for skip in [
                    "logger.py",
                    "contextlib",
                    "ipykernel",
                    "asyncio",
                    "runpy",
                    "model.py",
                ]
            ):
                lineno = outer.lineno
                break

        logger.info(
            f"Batch size: {batch_size}\nNum prompts to process: {len(inputs)}\nat {filename_short}:{lineno}"
        )

        if sort_by_length:
            dataset = BaseDataset(
                sorted(
                    inputs,
                    key=len,
                    reverse=True,
                )
            )
        else:
            dataset = BaseDataset(inputs)

        data_loader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            collate_fn=collator,
            shuffle=False,
        )

        with torch.no_grad():
            for _inputs, _model_inputs in tqdm(data_loader):
                for _ in range(10):  # for do_sample=True retries
                    try:
                        outputs = self.model_generate(_model_inputs, generation_kwargs)

                        yield BatchGenerationResult(
                            inputs=_inputs,
                            model_inputs=_model_inputs,
                            outputs=outputs,
                            generation_kwargs=generation_kwargs,
                        )
                        break
                    except torch.cuda.OutOfMemoryError as e:
                        raise e
                    except RuntimeError as e:
                        logger.error(
                            f"{e}\nsampling error. Trying again with smaller batch size"
                        )

    def batch_generate_dynamic(
        self,
        inputs: list[BaseInputs],
        generation_kwargs: dict,
        collator: BaseDataCollator,
        batch_size: int = 32,
    ) -> Generator[BatchGenerationResult, None, None]:
        assert isinstance(batch_size, int), "Starting batch size must be an integer"
        assert batch_size > 0, "Starting batch size must be greater than 0"
        assert check_power_of_2(batch_size), "Starting batch size must be a power of 2"

        # current batch size
        _batch_size = batch_size

        # avoid warnings
        if "do_sample" in generation_kwargs:
            if not generation_kwargs["do_sample"]:
                self.model.generation_config.temperature = None
                self.model.generation_config.top_p = None

        # set instance ids
        inputs = self.set_instance_ids(inputs)

        # keep track of pops from inputs when computation is done
        pop_offsets = torch.zeros(len(inputs), dtype=torch.int32)

        while True:  # error when out of memory
            try:
                batch_result_generator = self.batch_generate(
                    inputs=inputs,
                    generation_kwargs=generation_kwargs,
                    collator=collator,
                    batch_size=_batch_size,
                    sort_by_length=True,
                )

                for batch_id, batch_result in enumerate(batch_result_generator):
                    batch_result: BatchGenerationResult
                    for _input in batch_result.inputs:
                        # remove from inputs list
                        inputs.pop(_input.instance_id - pop_offsets[_input.instance_id])
                        # adjust pop offsets
                        pop_offsets[_input.instance_id + 1 :] += 1

                    batch_result.remove_input_instance_ids()

                    yield batch_result

                    if len(inputs) == 0:  # all done
                        batch_result_generator.close()
                        torch.cuda.empty_cache()
                        return

                    # Update batch size based on token counts
                    num_input_tokens: Tensor[BATCH] = batch_result.model_inputs[
                        "attention_mask"
                    ].sum(dim=-1)
                    if batch_id == 0:
                        max_tokens = int(num_input_tokens.max())

                    # Increase batch size if possible
                    if check_power_of_2(batch_size):
                        if int(num_input_tokens.min()) * 1.5 < max_tokens:
                            batch_size = int(batch_size * 1.5)
                            batch_result_generator.close()
                            break
                    elif int(num_input_tokens.min()) / 1.5 * 2 < max_tokens:
                        batch_size = int(batch_size / 1.5 * 2)
                        batch_result_generator.close()
                        break

            except torch.cuda.OutOfMemoryError:
                batch_result_generator.close()
                torch.cuda.empty_cache()

                # Decrease batch size
                if batch_size == 1:
                    raise Exception("Batch size is 1 and still out of memory") from None
                if check_power_of_2(batch_size):
                    batch_size = int(batch_size // 2 * 1.5)
                else:
                    batch_size = int(batch_size // 1.5)

    def run_inference(
        self,
        inputs: list[BaseInputs],
        generation_kwargs: dict,
        collator: BaseDataCollator,
        post_process_fn: callable,
        cache_file_name: str,
        batch_size: int = 32,
        dynamic_batch_sizing: bool = True,
        recompute: bool = False,
    ) -> Generator[BatchGenerationResult, None, None]:
        self.cache_reader = CacheReader(
            cache_jsonl_path=self.cache_jsonl_path,
        )

        inputs_to_compute = self.cache_reader.get_inputs_to_compute(
            inputs=inputs,
            generation_kwargs=generation_kwargs,
        )

        if self.cache_reader.precomputed_df.height > 0:
            for batch_result in self.cache_reader.get_batch_results(
                batch_size=batch_size
            ):
                yield batch_result

        if len(inputs_to_compute) == 0:
            return

        if self.model is None:
            self.load_model()

        batch_generate_fn = (
            self.batch_generate_dynamic if dynamic_batch_sizing else self.batch_generate
        )

        for batch_result in batch_generate_fn(
            generation_kwargs=generation_kwargs,
            collator=collator,
            batch_size=batch_size,
            inputs=inputs,
        ):
            post_processed = post_process_fn(batch_result)
            self.cache_reader.cache_results(post_processed)
            yield post_processed

class ModelCallWithCache(DeterministicModelWithCache):
    def model_generate(self, model_input_kwargs, generation_kwargs):
        model_input_kwargs = {
            k: v.to(self.model.device) if isinstance(v, torch.Tensor) else v
            for k, v in model_input_kwargs.items()
        }
        return self.model(**model_input_kwargs, **generation_kwargs)
