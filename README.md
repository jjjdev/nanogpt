# nanogpt
Nano GPT project - Build a GPT model following the paper "Attention is All You Need" and akarpathy's tutorial.

## Links and Resources
- Attention is All You Need paper: https://arxiv.org/abs/1706.03762
- OpenAI GPT-3 paper: https://arxiv.org/abs/2005.14165 

- akarpathy's google colab tutorial - https://colab.research.google.com/drive/1JMLa53HDuA-i7ZBmqV7ZnA3c_fvtXnx-?usp=sharing

- akarpathy's youtube tutorial - https://www.youtube.com/@AndrejKarpathy


## Overview
Building a lightweight language model (LLM) trained on a dataset that mimics Shakespeare's writing style.  We will learn how to train and fine-tune a GPT that will mimic the performance of OpenAI's GPT-2 (wow!).

The code is a very simple foundation for us to hack and improve, learn to train models from scratch, or finetune pretrained checkpoints.  

## Dataset
The dataset used for training NanoGPT consists of a collection of Shakespeare's plays, sonnets, and other literary works. This dataset is preprocessed to extract the text and convert it into a format suitable for training a language model.

## Model Architecture
NanoGPT utilizes a transformer-based architecture, which has proven to be highly effective in natural language processing tasks. The transformer model consists of multiple layers of self-attention and feed-forward neural networks. This architecture allows the model to capture long-range dependencies and generate coherent text.

## Training Process
During the training process, NanoGPT learns to predict the next word in a sequence given the previous words. This is achieved through a process called language modeling. The model is trained using a variant of the popular transformer-based training algorithm, such as the Transformer-XL or GPT-2.

## Evaluation Metrics
To evaluate the performance of NanoGPT, several metrics can be used, such as perplexity and BLEU score. Perplexity measures how well the model predicts the next word in a sequence, while BLEU score compares the generated text with reference text to assess its similarity.


## Usage
To use NanoGPT, follow the instructions in the project's documentation. This may involve installing the required dependencies, downloading the pre-trained model, and running the provided scripts or APIs to generate text.


## Credit
All credit goes to Andrei Karpathy!!  Please take a look at his incredible learning materials (links above somewhere) 