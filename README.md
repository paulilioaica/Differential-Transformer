# PyTorch Differential Transformer

## Overview

This repository provides a PyTorch implementation of the **Differential Transformer (Diff Transformer)**. 

The Diff Transformer enhances traditional transformer architectures by introducing **differential attention**, which helps reduce noise in attention mechanisms. This architecture is designed to improve long-context modeling, in-context learning, and reduce hallucination in large language models.

## Differential Transformer Architecture

The Diff Transformer builds on the original transformer design, with the following key innovations:

1. **Differential Attention Mechanism**: Unlike standard attention, which computes a single softmax over the input, differential attention subtracts two softmax distributions to suppress irrelevant information. This emphasizes more meaningful interactions between elements in the sequence, allowing for better focus on important inputs.

2. **Encoder-Decoder Structure**: Maintains the original encoder and decoder stacks with differential attention integrated, improving both context understanding and generation.

3. **Noise Reduction**: The subtraction mechanism reduces noise and enhances clarity in decision-making processes, improving performance in tasks like language modeling and retrieval.

For further details on the architecture, refer to the paper: [Differential Transformer](https://arxiv.org/pdf/2410.05258).

## Features

✨ **Noise-Resilient**: Improves accuracy by suppressing distractions through differential attention.

🧠 **Enhanced Attention**: Focuses more on relevant information, allowing for better handling of long-context and complex data.

## Setup

To get started with the Differential Transformer, clone the repository

```shell
git clone https://github.com/paulilioaica/PyTorch-DiffTransformer
cd PyTorch-DiffTransformer/src/
```