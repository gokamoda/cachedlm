"""AsyncGLM - Asynchronous Generative Language Model processing with queue management."""

import asyncio
import threading
import time
from typing import Any, Dict, List, Optional, Union
from queue import Queue, Full, Empty
import torch


class AsyncGLM:
    """
    Asynchronous wrapper for Hugging Face Generative Language Models.
    
    This class allows for asynchronous generation from language models while managing
    a queue to prevent buffer explosion. It supports batch inputs from transformers
    collators or Datasets.
    
    Args:
        model: A Hugging Face transformers model (from transformers.from_pretrained)
        tokenizer: Optional tokenizer for the model
        max_queue_size: Maximum size of the output queue (default: 10)
        generation_kwargs: Default kwargs to pass to model.generate()
    
    Example:
        >>> from transformers import AutoModelForCausalLM, AutoTokenizer
        >>> model = AutoModelForCausalLM.from_pretrained("gpt2")
        >>> tokenizer = AutoTokenizer.from_pretrained("gpt2")
        >>> async_glm = AsyncGLM(model, tokenizer, max_queue_size=5)
        >>> 
        >>> # Start processing
        >>> async_glm.start()
        >>> 
        >>> # Submit batches
        >>> input_ids = tokenizer("Hello world", return_tensors="pt").input_ids
        >>> task_id = async_glm.submit(input_ids)
        >>> 
        >>> # Get results (blocks until ready)
        >>> result = async_glm.get_result(task_id)
        >>> 
        >>> # Shutdown when done
        >>> async_glm.shutdown()
    """
    
    def __init__(
        self,
        model: Any,
        tokenizer: Optional[Any] = None,
        max_queue_size: int = 10,
        generation_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """Initialize the AsyncGLM."""
        self.model = model
        self.tokenizer = tokenizer
        self.max_queue_size = max_queue_size
        self.generation_kwargs = generation_kwargs or {}
        
        # Task management
        self._task_queue = Queue(maxsize=max_queue_size)
        self._results = {}
        self._next_task_id = 0
        self._task_id_lock = threading.Lock()
        
        # Worker thread
        self._worker_thread = None
        self._running = False
        self._shutdown_event = threading.Event()
    
    def start(self) -> None:
        """Start the background worker thread for processing tasks."""
        if self._running:
            return
        
        self._running = True
        self._shutdown_event.clear()
        self._worker_thread = threading.Thread(target=self._worker, daemon=True)
        self._worker_thread.start()
    
    def _worker(self) -> None:
        """Worker thread that processes tasks from the queue."""
        while self._running or not self._task_queue.empty():
            try:
                # Get task from queue with timeout
                task = self._task_queue.get(timeout=0.1)
                
                task_id = task["id"]
                inputs = task["inputs"]
                kwargs = task["kwargs"]
                
                try:
                    # Process the task
                    result = self._generate(inputs, kwargs)
                    self._results[task_id] = {"status": "success", "result": result}
                except Exception as e:
                    self._results[task_id] = {"status": "error", "error": str(e)}
                finally:
                    self._task_queue.task_done()
                    
            except Empty:
                # Queue timeout, check if we should continue
                if self._shutdown_event.is_set():
                    break
                continue
    
    def _generate(self, inputs: Any, kwargs: Dict[str, Any]) -> Any:
        """
        Generate output from the model.
        
        Args:
            inputs: Model inputs (can be tensors, dict, or batch)
            kwargs: Generation kwargs
            
        Returns:
            Generated output from the model
        """
        # Merge default kwargs with task-specific kwargs
        generation_params = {**self.generation_kwargs, **kwargs}
        
        # Handle different input types
        if isinstance(inputs, dict):
            # Dictionary input (e.g., from collator)
            with torch.no_grad():
                outputs = self.model.generate(**inputs, **generation_params)
        elif isinstance(inputs, torch.Tensor):
            # Tensor input
            with torch.no_grad():
                outputs = self.model.generate(inputs, **generation_params)
        else:
            # Assume it's already in the right format
            with torch.no_grad():
                outputs = self.model.generate(inputs, **generation_params)
        
        return outputs
    
    def submit(
        self,
        inputs: Union[torch.Tensor, Dict[str, torch.Tensor], Any],
        generation_kwargs: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
    ) -> int:
        """
        Submit a generation task to the queue.
        
        Args:
            inputs: Model inputs (can be batch from collator or Datasets)
            generation_kwargs: Optional kwargs to override defaults for this task
            timeout: Optional timeout for adding to queue (None = block indefinitely)
            
        Returns:
            Task ID that can be used to retrieve results
            
        Raises:
            RuntimeError: If AsyncGLM is not started
            Full: If queue is full and timeout expires
        """
        if not self._running:
            raise RuntimeError("AsyncGLM is not started. Call start() first.")
        
        # Get unique task ID
        with self._task_id_lock:
            task_id = self._next_task_id
            self._next_task_id += 1
        
        # Create task
        task = {
            "id": task_id,
            "inputs": inputs,
            "kwargs": generation_kwargs or {},
        }
        
        # Add to queue
        if timeout is None:
            self._task_queue.put(task)
        else:
            self._task_queue.put(task, timeout=timeout)
        
        return task_id
    
    def get_result(self, task_id: int, timeout: Optional[float] = None) -> Any:
        """
        Get the result of a submitted task.
        
        Args:
            task_id: The task ID returned by submit()
            timeout: Optional timeout in seconds (None = wait indefinitely)
            
        Returns:
            The generated output from the model
            
        Raises:
            TimeoutError: If timeout expires before result is ready
            RuntimeError: If task failed with an error
        """
        start_time = time.time()
        
        while True:
            if task_id in self._results:
                result_data = self._results.pop(task_id)
                
                if result_data["status"] == "error":
                    raise RuntimeError(f"Task {task_id} failed: {result_data['error']}")
                
                return result_data["result"]
            
            # Check timeout
            if timeout is not None:
                elapsed = time.time() - start_time
                if elapsed >= timeout:
                    raise TimeoutError(f"Timeout waiting for task {task_id}")
            
            # Sleep briefly before checking again
            time.sleep(0.01)
    
    def is_ready(self, task_id: int) -> bool:
        """
        Check if a task's result is ready.
        
        Args:
            task_id: The task ID to check
            
        Returns:
            True if result is ready, False otherwise
        """
        return task_id in self._results
    
    def queue_size(self) -> int:
        """
        Get the current size of the task queue.
        
        Returns:
            Number of tasks waiting in the queue
        """
        return self._task_queue.qsize()
    
    def shutdown(self, wait: bool = True, timeout: Optional[float] = None) -> None:
        """
        Shutdown the AsyncGLM and stop processing.
        
        Args:
            wait: If True, wait for all queued tasks to complete
            timeout: Optional timeout when waiting (None = wait indefinitely)
        """
        if not self._running:
            return
        
        self._running = False
        self._shutdown_event.set()
        
        if wait and self._worker_thread is not None:
            # Wait for queue to be processed
            self._task_queue.join()
            
            # Wait for worker thread to finish
            self._worker_thread.join(timeout=timeout)
    
    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.shutdown(wait=True)
        return False
    
    async def submit_async(
        self,
        inputs: Union[torch.Tensor, Dict[str, torch.Tensor], Any],
        generation_kwargs: Optional[Dict[str, Any]] = None,
    ) -> int:
        """
        Async version of submit - runs in executor to avoid blocking.
        
        Args:
            inputs: Model inputs (can be batch from collator or Datasets)
            generation_kwargs: Optional kwargs to override defaults for this task
            
        Returns:
            Task ID that can be used to retrieve results
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # Fall back for older Python versions or when not in async context
            loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.submit, inputs, generation_kwargs)
    
    async def get_result_async(self, task_id: int, timeout: Optional[float] = None) -> Any:
        """
        Async version of get_result.
        
        Args:
            task_id: The task ID returned by submit()
            timeout: Optional timeout in seconds
            
        Returns:
            The generated output from the model
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # Fall back for older Python versions or when not in async context
            loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.get_result, task_id, timeout)
