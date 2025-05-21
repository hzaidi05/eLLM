# eLLM: Educational LLM Inference Engine

A minimal, educational implementation of an LLM inference engine, forked from [Andrew K Chan's YALM](https://github.com/andrewkchan/yalm). This project aims to explore modern LLM inferencing strategies work under the hood.

## Key Features

### Core Components
- Transformer architecture implementation
- Multi-head attention with support for:
  - Multi-query attention (MQA)
  - Grouped-query attention (GQA)
- Rotary Position Embeddings (RoPE)
- RMSNorm layer normalization
- Mixture of Experts (MoE) support
- CUDA acceleration for GPU inference

### Implemented Kernels
- Efficient attention computation
- Matrix multiplication operations
- Layer normalization
- Activation functions (GELU, SiLU)
- KV-cache management
- CUDA graphs optimization!

## To-do

[] Better Top-K sampling implementation
[] Enhanced Top-P (nucleus) sampling
[] Speculative decoding
[] Flash Attention
[] Triton version?

## Requirements
- CUDA toolkit
- C++17 compatible compiler
- CMake 3.10+
