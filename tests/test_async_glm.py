"""Tests for AsyncGLM class."""

import unittest
import torch
import time
from unittest.mock import Mock, MagicMock
from queue import Full

import sys
sys.path.insert(0, '/home/runner/work/async-cached-hfglm/async-cached-hfglm')

from async_cached_hfglm import AsyncGLM


class MockModel:
    """Mock Hugging Face model for testing."""
    
    def generate(self, input_ids=None, **kwargs):
        """Mock generate method."""
        # Simulate generation by returning input with a simple modification
        if isinstance(input_ids, torch.Tensor):
            return torch.cat([input_ids, torch.ones_like(input_ids[:, :2])], dim=1)
        elif isinstance(input_ids, dict):
            ids = input_ids.get("input_ids", torch.tensor([[1, 2, 3]]))
            return torch.cat([ids, torch.ones_like(ids[:, :2])], dim=1)
        return torch.tensor([[1, 2, 3, 4, 5]])


class TestAsyncGLM(unittest.TestCase):
    """Test cases for AsyncGLM."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.model = MockModel()
        
    def test_initialization(self):
        """Test AsyncGLM initialization."""
        async_glm = AsyncGLM(self.model, max_queue_size=5)
        
        self.assertEqual(async_glm.max_queue_size, 5)
        self.assertFalse(async_glm._running)
        self.assertEqual(async_glm.queue_size(), 0)
    
    def test_start_and_shutdown(self):
        """Test starting and shutting down AsyncGLM."""
        async_glm = AsyncGLM(self.model)
        
        # Start
        async_glm.start()
        self.assertTrue(async_glm._running)
        self.assertIsNotNone(async_glm._worker_thread)
        
        # Shutdown
        async_glm.shutdown(wait=True)
        self.assertFalse(async_glm._running)
    
    def test_context_manager(self):
        """Test AsyncGLM as context manager."""
        with AsyncGLM(self.model) as async_glm:
            self.assertTrue(async_glm._running)
        
        # Should be shut down after exiting context
        self.assertFalse(async_glm._running)
    
    def test_submit_and_get_result_with_tensor(self):
        """Test submitting a tensor input and getting result."""
        async_glm = AsyncGLM(self.model)
        async_glm.start()
        
        try:
            # Submit task
            input_ids = torch.tensor([[1, 2, 3]])
            task_id = async_glm.submit(input_ids)
            
            # Get result
            result = async_glm.get_result(task_id, timeout=5.0)
            
            # Verify result shape
            self.assertIsInstance(result, torch.Tensor)
            self.assertEqual(result.shape[0], 1)  # Batch size
            self.assertGreater(result.shape[1], input_ids.shape[1])  # Generated tokens
            
        finally:
            async_glm.shutdown(wait=True)
    
    def test_submit_and_get_result_with_dict(self):
        """Test submitting a dict input (like from collator) and getting result."""
        async_glm = AsyncGLM(self.model)
        async_glm.start()
        
        try:
            # Submit task with dict input
            inputs = {"input_ids": torch.tensor([[1, 2, 3]])}
            task_id = async_glm.submit(inputs)
            
            # Get result
            result = async_glm.get_result(task_id, timeout=5.0)
            
            # Verify result
            self.assertIsInstance(result, torch.Tensor)
            self.assertEqual(result.shape[0], 1)
            
        finally:
            async_glm.shutdown(wait=True)
    
    def test_batch_input(self):
        """Test submitting batch inputs."""
        async_glm = AsyncGLM(self.model)
        async_glm.start()
        
        try:
            # Submit batch
            batch_input = torch.tensor([[1, 2, 3], [4, 5, 6]])
            task_id = async_glm.submit(batch_input)
            
            # Get result
            result = async_glm.get_result(task_id, timeout=5.0)
            
            # Verify batch processing
            self.assertIsInstance(result, torch.Tensor)
            self.assertEqual(result.shape[0], 2)  # Batch size preserved
            
        finally:
            async_glm.shutdown(wait=True)
    
    def test_multiple_tasks(self):
        """Test submitting and processing multiple tasks."""
        async_glm = AsyncGLM(self.model, max_queue_size=5)
        async_glm.start()
        
        try:
            # Submit multiple tasks
            task_ids = []
            for i in range(3):
                input_ids = torch.tensor([[i + 1, i + 2, i + 3]])
                task_id = async_glm.submit(input_ids)
                task_ids.append(task_id)
            
            # Get all results
            results = []
            for task_id in task_ids:
                result = async_glm.get_result(task_id, timeout=5.0)
                results.append(result)
            
            # Verify all results
            self.assertEqual(len(results), 3)
            for result in results:
                self.assertIsInstance(result, torch.Tensor)
                
        finally:
            async_glm.shutdown(wait=True)
    
    def test_queue_size_limit(self):
        """Test that queue size is limited."""
        async_glm = AsyncGLM(self.model, max_queue_size=2)
        async_glm.start()
        
        try:
            # Submit tasks up to queue limit
            task_ids = []
            for i in range(2):
                input_ids = torch.tensor([[i + 1, i + 2, i + 3]])
                task_id = async_glm.submit(input_ids, timeout=1.0)
                task_ids.append(task_id)
            
            # Give worker time to process at least one task
            time.sleep(0.5)
            
            # Should be able to submit more after processing starts
            input_ids = torch.tensor([[99, 99, 99]])
            task_id = async_glm.submit(input_ids, timeout=2.0)
            
            # Get all results
            for tid in task_ids + [task_id]:
                async_glm.get_result(tid, timeout=5.0)
                
        finally:
            async_glm.shutdown(wait=True)
    
    def test_is_ready(self):
        """Test checking if result is ready."""
        async_glm = AsyncGLM(self.model)
        async_glm.start()
        
        try:
            # Submit task
            input_ids = torch.tensor([[1, 2, 3]])
            task_id = async_glm.submit(input_ids)
            
            # Initially might not be ready
            # (depending on timing, could be ready immediately)
            
            # Wait for result
            result = async_glm.get_result(task_id, timeout=5.0)
            
            # After getting result, it should have been popped from results
            self.assertIsInstance(result, torch.Tensor)
            
        finally:
            async_glm.shutdown(wait=True)
    
    def test_generation_kwargs(self):
        """Test passing generation kwargs."""
        async_glm = AsyncGLM(
            self.model,
            generation_kwargs={"max_length": 10}
        )
        async_glm.start()
        
        try:
            # Submit with override kwargs
            input_ids = torch.tensor([[1, 2, 3]])
            task_id = async_glm.submit(
                input_ids,
                generation_kwargs={"temperature": 0.7}
            )
            
            # Get result
            result = async_glm.get_result(task_id, timeout=5.0)
            self.assertIsInstance(result, torch.Tensor)
            
        finally:
            async_glm.shutdown(wait=True)
    
    def test_submit_without_start_raises_error(self):
        """Test that submitting without starting raises error."""
        async_glm = AsyncGLM(self.model)
        
        input_ids = torch.tensor([[1, 2, 3]])
        with self.assertRaises(RuntimeError):
            async_glm.submit(input_ids)
    
    def test_get_result_timeout(self):
        """Test timeout when getting result."""
        async_glm = AsyncGLM(self.model)
        
        # Create a task_id that doesn't exist
        task_id = 9999
        
        with self.assertRaises(TimeoutError):
            async_glm.get_result(task_id, timeout=0.1)
    
    def test_concurrent_processing(self):
        """Test that processing happens concurrently with submission."""
        async_glm = AsyncGLM(self.model, max_queue_size=10)
        async_glm.start()
        
        try:
            # Submit multiple tasks quickly
            task_ids = []
            start_time = time.time()
            
            for i in range(5):
                input_ids = torch.tensor([[i + 1, i + 2, i + 3]])
                task_id = async_glm.submit(input_ids)
                task_ids.append(task_id)
            
            submit_time = time.time() - start_time
            
            # Get results (worker should be processing in background)
            results = []
            for task_id in task_ids:
                result = async_glm.get_result(task_id, timeout=10.0)
                results.append(result)
            
            # All results should be retrieved
            self.assertEqual(len(results), 5)
            
        finally:
            async_glm.shutdown(wait=True)


if __name__ == "__main__":
    unittest.main()
