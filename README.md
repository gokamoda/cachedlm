# cachedlm

## About
Dynamic batched runs for Hugging Face language models with caching support.  

### Dynamic Batching
Requests are ordered by input length and grouped into batches using dataloader.
Given starting batch size, this library dynamically reduces batchsize if GPU runs out of memory, and increases batchsize according to the reduction in model input length.

### Caching
Taking input arguments and model generation arguments as the cache key, this library caches model outputs to avoid redundant computations.


## Examples
See `src/x_*.py` files for example usages.
 


