"""Example using AsyncGLM with real Hugging Face transformers models."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from async_cached_hfglm import AsyncGLM


def with_real_model():
    """
    Example using a real transformer model.
    
    Note: This requires downloading the model, which can take time.
    Uncomment and run when you have the model downloaded.
    """
    print("Example with real transformers model:")
    print("(This would require actual model download)")
    print()
    
    # Uncomment to use with real model:
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    # Load a small model for testing
    model_name = "gpt2"
    print(f"Loading {model_name}...")
    
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Set padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Create AsyncGLM
    with AsyncGLM(model, tokenizer, max_queue_size=5) as async_glm:
        # Prepare inputs
        texts = [
            "Hello, my name is",
            "The future of AI is",
            "In a world where",
        ]
        
        print("Submitting generation tasks...")
        task_ids = []
        for text in texts:
            inputs = tokenizer(text, return_tensors="pt")
            task_id = async_glm.submit(
                inputs.input_ids,
                generation_kwargs={"max_length": 30, "do_sample": True}
            )
            task_ids.append((task_id, text))
            print(f"  Submitted: '{text}'")
        
        print("\nGenerating (this happens asynchronously)...")
        
        # Get and decode results
        print("\nResults:")
        for task_id, original_text in task_ids:
            result = async_glm.get_result(task_id, timeout=60.0)
            generated_text = tokenizer.decode(result[0], skip_special_tokens=True)
            print(f"\nOriginal: {original_text}")
            print(f"Generated: {generated_text}")
    
    print("\nDone!")
    """


def with_dataset_collator():
    """
    Example showing how to use AsyncGLM with DataLoader and collator.
    
    This demonstrates the typical use case with datasets and batch processing.
    """
    print("Example with Dataset and DataCollator pattern:")
    print("(Simulated with dummy data)")
    print()
    
    # Uncomment to use with real dataset:
    """
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        DataCollatorForLanguageModeling,
    )
    from torch.utils.data import DataLoader
    from datasets import load_dataset
    
    # Load model and tokenizer
    model_name = "gpt2"
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load and prepare dataset
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test[:10]")
    
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=128)
    
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    
    # Create data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    
    # Create DataLoader
    dataloader = DataLoader(
        tokenized_dataset,
        batch_size=2,
        collate_fn=data_collator,
    )
    
    # Use AsyncGLM with batches from DataLoader
    with AsyncGLM(model, tokenizer, max_queue_size=10) as async_glm:
        print("Processing batches from DataLoader...")
        
        task_ids = []
        for batch_idx, batch in enumerate(dataloader):
            # Submit batch for generation
            task_id = async_glm.submit(
                batch,  # Dict with 'input_ids', 'attention_mask', etc.
                generation_kwargs={"max_length": 150}
            )
            task_ids.append(task_id)
            print(f"  Submitted batch {batch_idx}")
            
            # Limit to first few batches for example
            if batch_idx >= 2:
                break
        
        print(f"\nProcessing {len(task_ids)} batches...")
        
        # Get results and do postprocessing
        all_results = []
        for idx, task_id in enumerate(task_ids):
            result = async_glm.get_result(task_id, timeout=120.0)
            print(f"  Got result for batch {idx}: shape {result.shape}")
            
            # Postprocessing example: decode first item in batch
            decoded = tokenizer.decode(result[0], skip_special_tokens=True)
            all_results.append(decoded[:100] + "...")  # Show first 100 chars
        
        print("\nSample outputs:")
        for idx, text in enumerate(all_results):
            print(f"Batch {idx}: {text}")
    
    print("\nDone!")
    """


def parallel_processing_pattern():
    """
    Demonstrate parallel processing pattern where postprocessing
    happens while generation is still running.
    """
    print("Parallel processing pattern (with dummy model):")
    print()
    
    # Create a dummy model for demonstration
    class DummyModel:
        def generate(self, input_ids, **kwargs):
            import time
            time.sleep(0.1)  # Simulate generation time
            if isinstance(input_ids, torch.Tensor):
                return torch.cat([input_ids, torch.ones_like(input_ids[:, :5])], dim=1)
            return input_ids
    
    model = DummyModel()
    
    with AsyncGLM(model, max_queue_size=10) as async_glm:
        import time
        
        # Submit several tasks
        print("Submitting 10 tasks...")
        task_ids = []
        for i in range(10):
            input_ids = torch.tensor([[i + 1, i + 2, i + 3]])
            task_id = async_glm.submit(input_ids)
            task_ids.append(task_id)
        
        print(f"Submitted all tasks. Queue size: {async_glm.queue_size()}")
        
        # Process results as they become available
        print("\nProcessing results as they complete...")
        processed_count = 0
        
        for task_id in task_ids:
            # Get result (blocks until ready)
            result = async_glm.get_result(task_id, timeout=10.0)
            
            # Simulate postprocessing
            time.sleep(0.05)
            processed = result.sum().item()
            processed_count += 1
            
            print(f"  Processed task {task_id}: sum={processed} ({processed_count}/{len(task_ids)})")
        
        print(f"\nCompleted processing all {processed_count} tasks!")


if __name__ == "__main__":
    print("AsyncGLM Transformers Integration Examples")
    print("=" * 60)
    print()
    
    with_real_model()
    print("\n" + "=" * 60 + "\n")
    
    with_dataset_collator()
    print("\n" + "=" * 60 + "\n")
    
    parallel_processing_pattern()
