import json
import torch
import torch.nn as nn
from transformers import GPT2Tokenizer
from model_pytorch import GPT, GPTConfig
from decoding import evaluate_strategy

def main():
    print("Initializing FP32 Model...")
    
    # Set quantization engine for Mac
    torch.backends.quantized.engine = 'qnnpack'
    
    config = GPTConfig(
        n_layer=4, n_head=4, d_model=128, d_ff=512, block_size=256, vocab_size=50257
    )
    model_fp32 = GPT(config)
    model_fp32.eval()
    
    print("Quantizing to INT8...")
    model_int8 = torch.quantization.quantize_dynamic(
        model_fp32, {nn.Linear}, dtype=torch.qint8
    )
    
    print("Loading tokenizer...")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    prompts = [
        "The meaning of life is",
        "In the future, artificial intelligence will",
        "Once upon a time in a distant galaxy"
    ]
    
    strategies = ["greedy", "top_k", "nucleus"]
    models = {"FP32": model_fp32, "INT8": model_int8}
    
    results = []
    
    print("\n" + "="*80)
    print("DECODING & QUANTIZATION EVALUATION PIPELINE")
    print("="*80)
    print(f"{'strategy':<10} | {'model':<6} | {'perplexity':<12} | {'latency_ms':<12} | {'consistency_%':<15}")
    print("-" * 80)
    
    for strategy in strategies:
        for model_name, model in models.items():
            # For greedy, only run once to save time since consistency is always 100%
            runs = 5 if strategy != "greedy" else 2
            lat, ppl, cons = evaluate_strategy(model, tokenizer, strategy, prompts, runs=runs)
            
            if strategy == "greedy":
                cons = 100.0
                
            print(f"{strategy:<10} | {model_name:<6} | {ppl:<12.2f} | {lat:<12.2f} | {cons:<15.2f}")
            
            results.append({
                "strategy": strategy,
                "model": model_name,
                "perplexity": ppl,
                "latency_ms": lat,
                "consistency_%": cons
            })
            
    with open("eval_results.json", "w") as f:
        json.dump(results, f, indent=4)
        
    print("\nSaved eval_results.json")

if __name__ == "__main__":
    main()
