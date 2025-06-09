import os, csv, json
import argparse
import time
from tqdm import tqdm
from datasets import load_dataset
import re
from openai import OpenAI
from transformers import AutoTokenizer
import tiktoken
import torch.multiprocessing as mp
from dotenv import load_dotenv
import hashlib
import pickle
from pathlib import Path

load_dotenv()

# Import Google AI for token counting
try:
    import google.generativeai as genai
    genai.configure(api_key=os.getenv("API_KEY"))
    GOOGLE_AI_AVAILABLE = True
except:
    GOOGLE_AI_AVAILABLE = False

# Set up cache directory
CACHE_DIR = Path(".cache/longbench_responses")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

model_map = json.loads(open('config/model2path.json', encoding='utf-8').read())
maxlen_map = json.loads(open('config/model2maxlen.json', encoding='utf-8').read())

# For Google Gemini: use https://generativelanguage.googleapis.com/v1beta/
# For local vLLM: use http://127.0.0.1:8000/v1
URL = os.getenv("API_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/")
API_KEY = os.getenv("API_KEY", "your-gemini-api-key-here")
template_rag = open('prompts/0shot_rag.txt', encoding='utf-8').read()
template_no_context = open('prompts/0shot_no_context.txt', encoding='utf-8').read()
template_0shot = open('prompts/0shot.txt', encoding='utf-8').read()
template_0shot_cot = open('prompts/0shot_cot.txt', encoding='utf-8').read()
template_0shot_cot_ans = open('prompts/0shot_cot_ans.txt', encoding='utf-8').read()

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

def query_llm(prompt, model, tokenizer, client=None, temperature=0.5, max_new_tokens=128, stop=None, use_cache=True, seed=None):
    # truncate
    max_len = maxlen_map[model]
    is_tiktoken = hasattr(tokenizer, '__module__') and 'tiktoken' in tokenizer.__module__
    
    # For Gemini models, use Google's token counting
    if "gemini" in model and GOOGLE_AI_AVAILABLE:
        gemini_model = genai.GenerativeModel(model_map[model])
        count_result = gemini_model.count_tokens(prompt)
        token_count = count_result.total_tokens
        
        if token_count > max_len:
            # Truncate at character level based on Google's ~4 chars/token
            chars_per_token = len(prompt) / token_count
            target_chars = int(max_len * 0.95 * chars_per_token)
            half = target_chars // 2
            prompt = prompt[:half] + prompt[-half:]
    elif model in model_map:
        if is_tiktoken:
            input_ids = tokenizer.encode(prompt, disallowed_special=())
        else:
            input_ids = tokenizer.encode(prompt)
        if len(input_ids) > max_len:
            input_ids = input_ids[:max_len//2] + input_ids[-max_len//2:]
            if is_tiktoken:
                prompt = tokenizer.decode(input_ids)
            else:
                prompt = tokenizer.decode(input_ids, skip_special_tokens=True)
    else:
        input_ids = tokenizer.encode(prompt, disallowed_special=())
        if len(input_ids) > max_len:
            input_ids = input_ids[:max_len//2] + input_ids[-max_len//2:]
            prompt = tokenizer.decode(input_ids)
    
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
    
    tries = 0
    if model in model_map:
        model = model_map[model]
    while tries < 5:
        tries += 1
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_new_tokens,
            )
            response = completion.choices[0].message.content
            
            # Save to cache
            if use_cache:
                try:
                    with open(cache_file, 'wb') as f:
                        pickle.dump(response, f)
                except:
                    pass  # If cache write fails, continue
                    
            return response
        except KeyboardInterrupt as e:
            raise e
        except Exception as e:
            error_msg = str(e)
            print(f"Error Occurs: \"{error_msg}\"        Retry ...")
            
            # Check for rate limit error and use suggested retry delay
            if "429" in error_msg and "retryDelay" in error_msg:
                import re
                delay_match = re.search(r"'retryDelay': '(\d+)s'", error_msg)
                if delay_match:
                    delay = int(delay_match.group(1))
                    print(f"Rate limited - waiting {delay} seconds as suggested by API")
                    time.sleep(delay)
                else:
                    time.sleep(min(2 ** tries, 60))  # Exponential backoff up to 60s
            else:
                time.sleep(min(2 ** tries, 60))  # Exponential backoff up to 60s
    else:
        print("Max tries. Failed.")
        return ''

def extract_answer(response):
    response = response.replace('*', '')
    match = re.search(r'The correct answer is \(([A-D])\)', response)
    if match:
        return match.group(1)
    else:
        match = re.search(r'The correct answer is ([A-D])', response)
        if match:
            return match.group(1)
        else:
            return None

def get_pred(data, args, out_file, rank):
    model = args.model
    if "gpt" in model or "o1" in model or "gemini" in model:
        tokenizer = tiktoken.encoding_for_model("gpt-4o-2024-08-06")
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_map[model], trust_remote_code=True)
    client = OpenAI(
        base_url=URL,
        api_key=API_KEY
    )
    # Open file handle for this process
    fout = open(out_file, 'a', encoding='utf-8')
    
    for item in tqdm(data, desc=f"Process {rank}"):
        context = item['context']
        if args.rag > 0:
            template = template_rag
            retrieved = item["retrieved_context"][:args.rag]
            retrieved = sorted(retrieved, key=lambda x: x['c_idx'])
            context = '\n\n'.join([f"Retrieved chunk {idx+1}: {x['content']}" for idx, x in enumerate(retrieved)])
        elif args.no_context:
            template = template_no_context
        elif args.cot:
            template = template_0shot_cot
        else:
            template = template_0shot
        prompt = template.replace('$DOC$', context.strip()).replace('$Q$', item['question'].strip()).replace('$C_A$', item['choice_A'].strip()).replace('$C_B$', item['choice_B'].strip()).replace('$C_C$', item['choice_C'].strip()).replace('$C_D$', item['choice_D'].strip())
        
        if args.cot:
            output = query_llm(prompt, model, tokenizer, client, temperature=0.1, max_new_tokens=1024, 
                             use_cache=not args.no_cache, seed=args.seed)
        else:
            output = query_llm(prompt, model, tokenizer, client, temperature=0.1, max_new_tokens=128,
                             use_cache=not args.no_cache, seed=args.seed)
        if output == '':
            continue
        if args.cot: # extract answer
            response = output.strip()
            item['response_cot'] = response
            prompt = template_0shot_cot_ans.replace('$DOC$', context.strip()).replace('$Q$', item['question'].strip()).replace('$C_A$', item['choice_A'].strip()).replace('$C_B$', item['choice_B'].strip()).replace('$C_C$', item['choice_C'].strip()).replace('$C_D$', item['choice_D'].strip()).replace('$COT$', response)
            output = query_llm(prompt, model, tokenizer, client, temperature=0.1, max_new_tokens=128,
                             use_cache=not args.no_cache, seed=args.seed)
            if output == '':
                continue
        response = output.strip()
        item['response'] = response
        item['pred'] = extract_answer(response)
        item['judge'] = item['pred'] == item['answer']
        item['context'] = context[:1000]
        fout.write(json.dumps(item, ensure_ascii=False) + '\n')
        fout.flush()
    fout.close()

def main():
    os.makedirs(args.save_dir, exist_ok=True)
    print(args)
    if args.rag > 0:
        out_file = os.path.join(args.save_dir, args.model.split("/")[-1] + f"_rag_{str(args.rag)}.jsonl")
    elif args.no_context:
        out_file = os.path.join(args.save_dir, args.model.split("/")[-1] + "_no_context.jsonl")
    elif args.cot:
        out_file = os.path.join(args.save_dir, args.model.split("/")[-1] + "_cot.jsonl")
    else:
        out_file = os.path.join(args.save_dir, args.model.split("/")[-1] + ".jsonl")

    dataset = load_dataset('THUDM/LongBench-v2', split='train') # dataset = json.load(open('data.json', 'r', encoding='utf-8'))
    data_all = [{"_id": item["_id"], "domain": item["domain"], "sub_domain": item["sub_domain"], "difficulty": item["difficulty"], "length": item["length"], "question": item["question"], "choice_A": item["choice_A"], "choice_B": item["choice_B"], "choice_C": item["choice_C"], "choice_D": item["choice_D"], "answer": item["answer"], "context": item["context"]} for item in dataset]

    # cache
    has_data = {}
    if os.path.exists(out_file):
        with open(out_file, encoding='utf-8') as f:
            has_data = {json.loads(line)["_id"]: 0 for line in f}
    data = []
    for item in data_all:
        if item["_id"] not in has_data:
            data.append(item)

    data_subsets = [data[i::args.n_proc] for i in range(args.n_proc)]
    processes = []
    for rank in range(args.n_proc):
        p = mp.Process(target=get_pred, args=(data_subsets[rank], args, out_file, rank))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", "-s", type=str, default="results")
    parser.add_argument("--model", "-m", type=str, default="GLM-4-9B-Chat")
    parser.add_argument("--cot", "-cot", action='store_true') # set to True if using COT
    parser.add_argument("--no_context", "-nc", action='store_true') # set to True if using no context (directly measuring memorization)
    parser.add_argument("--rag", "-rag", type=int, default=0) # set to 0 if RAG is not used, otherwise set to N when using top-N retrieved context
    parser.add_argument("--n_proc", "-n", type=int, default=16)
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducible results")
    parser.add_argument("--no_cache", action='store_true', help="Disable response caching")
    args = parser.parse_args()
    main()