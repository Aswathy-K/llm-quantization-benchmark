# LLM Quantization Benchmark: TinyLlama

Benchmarking INT8 and INT4 quantization methods on TinyLlama using WikiText-2 
perplexity as the accuracy metric. Covers bitsandbytes LLM.int8() and 
AutoAWQ with INT4 GEMM kernels.

## Results

| Method                   | Perplexity ↓ | Size (MB) | Compression |
|---------------------------------|-------------|-----------|-------------|
| FP16 baseline                   | 10.24       | 2200 MB    | 1.0x        |
| INT8 (bnb)                      | 10.28       | 1231 MB    | 1.79x        |
| INT4 W4A16(llm-compressor)      | 11.37       | 762 MB     | 2.88x        |

![Benchmark chart](results/benchmark_chart.png)

## Key findings

**INT8** preserves accuracy almost perfectly (+0.04 PPL) with 1.8x compression.
The LLM.int8() mixed-precision approach keeps outlier activation channels in 
FP16, which explains the near-lossless quality. Memory reduction matters more 
than compute speedup here — INT8 weights still dequantize to FP16 for matmul.

**AWQ INT4** achieves 2.88x compression with a modest accuracy cost (+1.13 PPL).
The group_size=128 config balances accuracy vs overhead well. At this 
compression level, a 70B model would fit in ~48GB instead of ~140GB — 
enabling 2xA100 serving instead of requiring 4x.

Theoretical INT4 compression is 4x vs FP16. Actual compression is 2.88x because grouped quantization stores FP16 scale factors per 128-weight group, adding ~33% overhead. This is the standard tradeoff — smaller groups give better accuracy but more scale overhead.

## Quantization method comparison

| | INT8 (LLM.int8()) | AWQ INT4 |
|---|---|---|
| Weight precision | INT8 | INT4 |
| Activation precision | FP16 (mixed) | FP16 |
| Compute savings | Memory only | Memory only |
| Best for | Quality-sensitive | Throughput / deployment |
| Calibration needed | No | Yes (~128 samples) |

<img width="1089" height="390" alt="image" src="https://github.com/user-attachments/assets/29ee1fc0-c628-4684-b254-aed76e3b2803" />



Tested on: Google Colab T4 (16GB VRAM), Python 3.10, PyTorch 2.2
