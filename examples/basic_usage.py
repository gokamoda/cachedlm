"""Basic usage example for AsyncGLM."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from async_cached_hfglm import AsyncGLM


class DummyModel:
    """Dummy model for demonstration (replace with real model)."""
    
    def generate(self, input_ids, **kwargs):
        """Mock generation - just echoes input with extra tokens."""
        if isinstance(input_ids, torch.Tensor):
            # Simulate generation by adding tokens
            return torch.cat([input_ids, torch.ones_like(input_ids[:, :5])], dim=1)
        return input_ids


def main():
    """Demonstrate basic AsyncGLM usage."""
    
    # In real usage, load a real model:
    # from transformers import AutoModelForCausalLM, AutoTokenizer
    # model = AutoModelForCausalLM.from_pretrained("gpt2")
    # tokenizer = AutoTokenizer.from_pretrained("gpt2")
    
    # For demo, use dummy model
    model = DummyModel()
    
    # Create AsyncGLM with queue size limit
    async_glm = AsyncGLM(
        model,
        max_queue_size=5,
        generation_kwargs={"max_length": 50}
    )
    
    # Start processing
    async_glm.start()
    
    try:
        print("Submitting tasks...")
        
        # Submit multiple generation tasks
        task_ids = []
        for i in range(3):
            # Create sample input
            input_ids = torch.tensor([[1 + i, 2 + i, 3 + i]])
            
            # Submit task
            task_id = async_glm.submit(input_ids)
            task_ids.append(task_id)
            print(f"Submitted task {task_id}")
        
        print(f"\nQueue size: {async_glm.queue_size()}")
        
        # Do some postprocessing while generation happens in background
        print("\nDoing postprocessing while generation runs...")
        processed_results = []
        
        for task_id in task_ids:
            # Check if ready (non-blocking)
            if async_glm.is_ready(task_id):
                print(f"Task {task_id} is already ready!")
            
            # Get result (blocking until ready)
            result = async_glm.get_result(task_id, timeout=10.0)
            print(f"Got result for task {task_id}: shape {result.shape}")
            
            # Do postprocessing
            processed = result * 2  # Example postprocessing
            processed_results.append(processed)
        
        print(f"\nProcessed {len(processed_results)} results")
        
    finally:
        # Shutdown and wait for remaining tasks
        print("\nShutting down...")
        async_glm.shutdown(wait=True)
        print("Done!")


def batch_example():
    """Demonstrate batch processing."""
    model = DummyModel()
    
    # Use context manager for automatic cleanup
    with AsyncGLM(model, max_queue_size=10) as async_glm:
        print("Batch processing example...")
        
        # Submit a batch (like from a DataLoader)
        batch_inputs = torch.tensor([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
        ])
        
        task_id = async_glm.submit(batch_inputs)
        print(f"Submitted batch task {task_id}")
        
        # Get batch result
        batch_result = async_glm.get_result(task_id, timeout=10.0)
        print(f"Got batch result: shape {batch_result.shape}")
        print("Batch processing complete!")


def dict_input_example():
    """Demonstrate dictionary input (like from collator)."""
    model = DummyModel()
    
    with AsyncGLM(model) as async_glm:
        print("Dictionary input example...")
        
        # Submit dict input (like from DataCollator)
        inputs = {
            "input_ids": torch.tensor([[1, 2, 3, 4]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1]]),
        }
        
        task_id = async_glm.submit(inputs)
        result = async_glm.get_result(task_id, timeout=10.0)
        print(f"Got result from dict input: shape {result.shape}")


if __name__ == "__main__":
    main()
    print("\n" + "="*50 + "\n")
    batch_example()
    print("\n" + "="*50 + "\n")
    dict_input_example()
