# Pytorch Transformer


## Overview
This is a PyTorch project that implements a plain transformer architecture for self-supervised prediction, which is at the core of LLMs. This project aims to provide a simple and efficient implementation of the transformer model, allowing users to train their own models for various tasks.

## Transformer Architecture

![Transformer Architecture Diagram](https://machinelearningmastery.com/wp-content/uploads/2021/08/attention_research_1-500x1536.png)

 The transformer architecture consists of the following components:

1. **Encoder**: The encoder takes in the input sequence and processes it through a stack of encoder layers. Each encoder layer consists of a multi-head self-attention mechanism and a feed-forward neural network.

2. **Decoder**: The decoder takes in the output of the encoder and generates the final output sequence. It also consists of a stack of decoder layers. Each decoder layer has a multi-head self-attention mechanism, a multi-head attention mechanism over the encoder outputs, and a feed-forward neural network.

3. **Attention Mechanism**: The attention mechanism allows the model to focus on different parts of the input sequence during the encoding and decoding process. It computes a weighted sum of the input sequence based on the relevance of each element to the current position.

4. **Feed-Forward Neural Network**: The feed-forward neural network is a fully connected layer that applies non-linear transformations to the input sequence.



For more details on the transformer architecture, refer to the original paper: [Attention Is All You Need](https://arxiv.org/abs/1706.03762).



## Features

✨ Easy-to-use: The project provides a straightforward setup and training loop for self-supervised prediction.

🧠 Transformer Architecture: The project implements the popular transformer architecture, which has shown great success in various natural language processing and computer vision tasks.

🔀 Self-Supervised Prediction: The training loop is designed to support self-supervised prediction, enabling the model to learn from unlabeled data.

## Setup

To get started with Transformer Plain, follow these steps:

1. Clone the repository:

    ```shell
    git clone https://github.com/paulilioaica/Pytorch-Transformer
    cd Pytorch-Transformer/src/

    ```

2. Install the required dependencies:

    ```shell
    pip install -r requirements.txt
    ```

## Usage

1. Dataset: Make sure you have a dataset suitable for self-supervised prediction from Huggingface (or use the AG-NEWS one). Simply pass the `dataset_name` for training on your dataset of choice.

2. Configure the training parameters: Adjust the hyperparameters by passing your own arguments.

3. Train the model: Run the training script to start the self-supervised prediction training loop.

4. Evaluate the model: Use the trained model to make predictions on your test dataset and evaluate its performance.

## Example run
```
python main.py  --num_layers 2 --n_heads 8 --seq_len 128 --num_epochs 10 --batch_size 32 --lr 0.001 --device cpu --embedding_dim 128 --dataset_name ag_news
```

## Results after 10 epochs
```
Epoch: 0, Loss: 10.326
Epoch: 1, Loss: 7.256
Epoch: 2, Loss: 5.972
...
Epoch: 9, Loss: 3.922

Test loss: 3.949
```

## License

This project is licensed under the MIT License. 
