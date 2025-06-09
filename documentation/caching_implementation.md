# LongBench Response Caching Implementation

## Overview

This document describes the deterministic response caching system implemented for the LongBench evaluation framework. The caching system ensures reproducible results, reduces API costs, and enables consistent benchmarking across multiple runs.

## Design Principles

### 1. Deterministic Key Generation

The cache key is generated using SHA-256 hashing of the following parameters:
- **prompt**: The full prompt sent to the model (after truncation)
- **model**: The model identifier (e.g., "gemini-2.0-flash", "gpt-4")
- **temperature**: The temperature parameter for generation
- **max_new_tokens**: Maximum tokens to generate
- **seed**: Random seed for reproducibility (defaults to "default" if not specified)

### 2. Cache Key Implementation

```python
def get_cache_key(prompt, model, temperature, max_new_tokens, seed=None):
    """Generate deterministic cache key for LLM queries"""
    key_data = {
        'prompt': prompt,
        'model': model,
        'temperature': temperature,
        'max_new_tokens': max_new_tokens,
        'seed': seed if seed is not None else 'default'
    }
    key_str = json.dumps(key_data, sort_keys=True)
    return hashlib.sha256(key_str.encode()).hexdigest()
```

The use of `sort_keys=True` ensures consistent key generation regardless of dictionary ordering.

## Implementation Details

### 1. Cache Storage

- **Location**: `.cache/longbench_responses/`
- **Format**: Pickle files named `{cache_key}.pkl`
- **Content**: Raw response string from the LLM
- **Creation**: Directory created automatically on first run

### 2. Cache Integration in query_llm

The caching logic is integrated into the `query_llm` function:

```python
# Check cache first
if use_cache:
    cache_key = get_cache_key(prompt, model, temperature, max_new_tokens, seed)
    cache_file = CACHE_DIR / f"{cache_key}.pkl"
    
    if cache_file.exists():
        try:
            with open(cache_file, 'rb') as f:
                cached_response = pickle.load(f)
                print(f"  [Cache hit for {model}]")
                return cached_response
        except:
            pass  # If cache read fails, continue to API call

# ... API call happens here ...

# Save to cache
if use_cache:
    try:
        with open(cache_file, 'wb') as f:
            pickle.dump(response, f)
    except:
        pass  # If cache write fails, continue
```

### 3. Command Line Arguments

Two new arguments were added:

```python
parser.add_argument("--seed", type=int, default=None, 
                    help="Random seed for reproducible results")
parser.add_argument("--no_cache", action='store_true', 
                    help="Disable response caching")
```

### 4. Cache Behavior by Model

**Important**: Each model has its own cache namespace. This means:
- `gemini-2.0-flash` and `gemini-1.5-pro` will NOT share cached responses
- `gpt-4` and `gpt-3.5-turbo` will NOT share cached responses
- This is intentional as different models produce different outputs

## Usage Examples

### Basic Usage with Caching

```bash
# First run - makes API calls and caches responses
python pred.py -m gemini-2.0-flash --seed 42

# Second run - uses cached responses (no API calls)
python pred.py -m gemini-2.0-flash --seed 42
```

### Different Seeds

```bash
# Seed 42 - one set of responses
python pred.py -m gemini-2.0-flash --seed 42

# Seed 123 - different responses (new API calls)
python pred.py -m gemini-2.0-flash --seed 123

# Back to seed 42 - uses cache from first run
python pred.py -m gemini-2.0-flash --seed 42
```

### Different Models

```bash
# Gemini model - creates its own cache
python pred.py -m gemini-2.0-flash --seed 42

# GPT-4 - separate cache, new API calls
python pred.py -m gpt-4 --seed 42

# Each model maintains independent cache
```

### Disabling Cache

```bash
# Force fresh API calls
python pred.py -m gemini-2.0-flash --no_cache
```

## Benefits

### 1. Cost Savings

- Re-running evaluations with same parameters costs $0
- Testing different evaluation metrics on same responses is free
- Debugging and development iterations don't incur API costs

### 2. Reproducibility

- Same seed guarantees same results across runs
- Enables consistent benchmarking
- Facilitates debugging and result verification

### 3. Performance

- Cached responses return instantly
- No network latency or API rate limiting for cached queries
- Enables rapid iteration on evaluation logic

### 4. Reliability

- Resilient to API outages for cached responses
- Graceful fallback if cache read/write fails
- No dependency on cache availability

## Technical Considerations

### 1. Cache Invalidation

The cache has no automatic invalidation. To force fresh results:
- Use `--no_cache` flag
- Change the seed
- Delete `.cache/longbench_responses/` directory
- Modify any parameter that affects the cache key

### 2. Storage Requirements

- Each cached response is typically 100-1000 bytes
- Full evaluation (503 examples) ≈ 500KB
- Multiple seeds/models multiply storage linearly

### 3. Truncation and Caching

**Important**: The cache key is generated AFTER prompt truncation. This means:
- Same long prompt truncated to fit different models may have different cache keys
- Gemini's token-based truncation vs tiktoken truncation creates different prompts
- This is correct behavior as truncated prompts may differ

### 4. Temperature Considerations

Even with `temperature=0.1`, responses may vary slightly between API calls. The cache ensures perfect reproducibility by storing the first response and reusing it.

## Security Notes

1. **No sensitive data**: Cache contains only model responses, not API keys
2. **Local storage**: Cache is stored locally, not uploaded anywhere
3. **Gitignored**: Add `.cache/` to `.gitignore` to prevent accidental commits

## Future Enhancements

Potential improvements to consider:

1. **Compression**: Use gzip to reduce cache storage
2. **Expiration**: Add timestamp-based cache expiration
3. **Cache stats**: Add command to show cache size and hit rates
4. **Shared cache**: Support for team-shared cache via cloud storage
5. **Cache warming**: Pre-populate cache for common evaluations

## Conclusion

The caching implementation provides a robust, deterministic system for storing and reusing LLM responses. It significantly reduces evaluation costs while ensuring reproducible results, making it ideal for benchmarking and research applications.