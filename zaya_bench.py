import argparse
import torch
import time
import json
import sys

def benchmark_gemm(M, K, N, dtype=torch.float16, warmup=3, iters=20):
    """Замер TFLOPS для конкретной тройки размеров матриц."""
    try:
        A = torch.empty(M, K, device='cuda', dtype=dtype).normal_(mean=0, std=0.02)
        B = torch.empty(K, N, device='cuda', dtype=dtype).normal_(mean=0, std=0.02)
    except Exception:
        return 0.0
        
    # Warmup
    for _ in range(warmup):
        torch.mm(A, B)
    torch.cuda.synchronize()
    
    start = time.time()
    for _ in range(iters):
        torch.mm(A, B)
    torch.cuda.synchronize()
    end = time.time()
    
    flops = 2.0 * M * K * N * iters
    tflops = flops / (end - start) / 1e12
    return tflops

def main():
    parser = argparse.ArgumentParser(description="ZAYA Hardware-Aware GEMM Benchmark")
    parser.add_argument("--out", default="zaya_bench_results.json")
    parser.add_argument("--dtype", default="fp16", choices=["fp16", "fp32"])
    parser.add_argument("--iters", type=int, default=20)
    
    # Архитектурные константы ZAYA
    parser.add_argument("--dims", type=int, nargs="+", default=[256, 320, 384, 512, 640, 768, 1024, 1280])
    parser.add_argument("--m-sizes", type=int, nargs="+", default=[256, 512, 1024, 2048, 4096, 8192]) # batch * seq
    parser.add_argument("--router-dim", type=int, default=192)
    parser.add_argument("--vocab-size", type=int, default=16384)
    parser.add_argument("--cca-comp", type=int, default=4)
    parser.add_argument("--expert-mult", type=float, default=1.3)
    parser.add_argument("--experts", type=int, default=8)
    
    args = parser.parse_args()
    dtype = torch.float16 if args.dtype == "fp16" else torch.float32
    
    shapes = set()
    for dim in args.dims:
        latent = dim // args.cca_comp
        # Квантование hidden (как в рекомендациях, кратно 64)
        hidden = max(64, round(dim * args.expert_mult / 64) * 64)
        
        for M in args.m_sizes:
            shapes.add((M, dim, latent * 3))           # Attn QKV
            shapes.add((M, latent, dim))               # Attn Out
            shapes.add((M, dim, args.router_dim))      # MoE Router Down
            shapes.add((M, args.router_dim, args.experts)) # MoE Router Up
            shapes.add((M, dim, hidden))               # MoE Expert FC1
            shapes.add((M, hidden, dim))               # MoE Expert FC2
            shapes.add((M, dim, args.vocab_size))      # LM Head
            
    shapes = list(shapes)
    total = len(shapes)
    
    print(f"Starting ZAYA benchmark for {total} specific architectural shapes...")
    results = {}
    
    for i, (M, K, N) in enumerate(shapes):
        tflops = benchmark_gemm(M, K, N, dtype=dtype, iters=args.iters)
        key = f"{M}_{K}_{N}"
        results[key] = round(tflops, 2)
        
        if (i + 1) % 10 == 0 or (i + 1) == total:
            print(f"\rProgress: {i+1}/{total} ({100*(i+1)/total:.1f}%) | Last: {key} -> {tflops:.2f} TFLOPS", end="", flush=True)
            
    print("\nBenchmark complete. Saving...")
    with open(args.out, "w") as f:
        json.dump({"dtype": args.dtype, "shapes": results}, f, indent=2)
    print(f"Saved hardware profile to {args.out}")

if __name__ == "__main__":
    main()
