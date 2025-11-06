# async-cached-hfglm

Asynchronous Generative Language Model processing with queue management for Hugging Face Transformers.

## Overview

`AsyncGLM` is a Python library that provides asynchronous processing for Hugging Face generative language models. It allows you to submit generation tasks to a queue and retrieve results asynchronously, enabling efficient parallel processing where the parent script can perform postprocessing while the model generates outputs in the background.

### Key Features

- **Asynchronous Processing**: Submit generation tasks and continue with other work while the model processes in the background
- **Queue Management**: Built-in queue with configurable size limits to prevent buffer explosion
- **Batch Support**: Compatible with batch inputs from transformers collators or Datasets
- **Thread-Safe**: Safe to use from multiple threads
- **Context Manager Support**: Easy resource management with context managers
- **Flexible Input**: Supports tensor inputs, dictionary inputs (from collators), and batch inputs

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from async_cached_hfglm import AsyncGLM

# Load your model
model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# Create AsyncGLM with queue size limit
async_glm = AsyncGLM(model, tokenizer, max_queue_size=5)

# Start processing
async_glm.start()

# Submit generation tasks
input_ids = tokenizer("Hello world", return_tensors="pt").input_ids
task_id = async_glm.submit(input_ids)

# Do other work while generation happens...
# ...

# Get result when ready
result = async_glm.get_result(task_id)
generated_text = tokenizer.decode(result[0])

# Shutdown when done
async_glm.shutdown()
```

## Usage

### Basic Usage

```python
from async_cached_hfglm import AsyncGLM

# Initialize with your model
async_glm = AsyncGLM(
    model,
    tokenizer=tokenizer,
    max_queue_size=10,
    generation_kwargs={"max_length": 50, "temperature": 0.7}
)

# Start the background worker
async_glm.start()

# Submit tasks
task_id = async_glm.submit(input_ids)

# Get results
result = async_glm.get_result(task_id, timeout=30.0)

# Clean up
async_glm.shutdown(wait=True)
```

### Context Manager

```python
with AsyncGLM(model, max_queue_size=5) as async_glm:
    task_id = async_glm.submit(input_ids)
    result = async_glm.get_result(task_id)
    # Automatically shuts down when exiting context
```

### Batch Processing

```python
# Submit a batch (e.g., from DataLoader)
batch_inputs = torch.tensor([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9],
])

task_id = async_glm.submit(batch_inputs)
batch_results = async_glm.get_result(task_id)
```

### Dictionary Inputs (from Collators)

```python
# Submit dict input (like from DataCollatorWithPadding)
inputs = {
    "input_ids": torch.tensor([[1, 2, 3, 4]]),
    "attention_mask": torch.tensor([[1, 1, 1, 1]]),
}

task_id = async_glm.submit(inputs)
result = async_glm.get_result(task_id)
```

### Multiple Tasks with Postprocessing

```python
with AsyncGLM(model, max_queue_size=10) as async_glm:
    # Submit multiple tasks
    task_ids = []
    for text in texts:
        input_ids = tokenizer(text, return_tensors="pt").input_ids
        task_id = async_glm.submit(input_ids)
        task_ids.append(task_id)
    
    # Process results as they become available
    for task_id in task_ids:
        # Check if ready (non-blocking)
        if async_glm.is_ready(task_id):
            result = async_glm.get_result(task_id)
            # Do postprocessing...
        else:
            # Do other work...
            pass
```

## API Reference

### AsyncGLM

#### `__init__(model, tokenizer=None, max_queue_size=10, generation_kwargs=None)`

Initialize AsyncGLM.

**Parameters:**
- `model`: Hugging Face transformers model (from `transformers.from_pretrained`)
- `tokenizer`: Optional tokenizer for the model
- `max_queue_size`: Maximum number of tasks in queue (default: 10)
- `generation_kwargs`: Default kwargs passed to `model.generate()`

#### `start()`

Start the background worker thread for processing tasks.

#### `submit(inputs, generation_kwargs=None, timeout=None)`

Submit a generation task to the queue.

**Parameters:**
- `inputs`: Model inputs (tensor, dict, or batch)
- `generation_kwargs`: Optional kwargs to override defaults
- `timeout`: Optional timeout for adding to queue

**Returns:** Task ID (integer)

#### `get_result(task_id, timeout=None)`

Get the result of a submitted task.

**Parameters:**
- `task_id`: Task ID returned by `submit()`
- `timeout`: Optional timeout in seconds

**Returns:** Generated output from the model

#### `is_ready(task_id)`

Check if a task's result is ready (non-blocking).

**Returns:** Boolean

#### `queue_size()`

Get the current number of tasks waiting in the queue.

**Returns:** Integer

#### `shutdown(wait=True, timeout=None)`

Shutdown AsyncGLM and stop processing.

**Parameters:**
- `wait`: If True, wait for all queued tasks to complete
- `timeout`: Optional timeout when waiting

## Examples

See the `examples/` directory for more detailed examples:
- `basic_usage.py`: Basic usage patterns and examples

## Requirements

- Python >= 3.7
- PyTorch >= 1.9.0
- Transformers >= 4.20.0

## Testing

Run tests with:

```bash
python -m pytest tests/
```

Or with unittest:

```bash
python -m unittest discover tests/
```

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.