
# English to Italian translator using Transformer

This repository contains code and utilities for training a english to italian translator using a transformer implemented on PyTorch Lightning framework. The code used for the transformers is similar to the original Transformer paper: ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762) by Vaswani et al.

## File Descriptions

- `lightning_module.py`: Contains code related to the PyTorch Lightning framework, including callback definitions.
- `train.py`: Core file for training the model. It handles dataset loading, model training, and utilities for progress tracking.
- `model.py`: Defines the neural network architectures, including custom layers and blocks.
- `dataset.py`: Handles dataset processing, specifically for bilingual datasets, which could be used in machine translation tasks.
- `config.py`: Provides configurations for the training process, including batch size, epochs, learning rate, and other essential parameters.

### [config.py](https://github.com/mkthoma/custom_transformer/blob/main/lib/config.py)

- get_config(): This function returns a dictionary that holds various configuration settings for your model and training process. These settings include parameters like batch size, number of epochs, learning rate, sequence length, model dimensions, source and target languages, model file paths, and more.

- get_weights_file_path(config, epoch): This function takes the configuration dictionary and an epoch as input, and it generates the file path for saving/loading model weights. It combines the model folder, model basename, and epoch number to create a complete file path.

### [dataset.py](https://github.com/mkthoma/custom_transformer/blob/main/lib/dataset.py)

 This dataset class is designed to handle bilingual data for a sequence-to-sequence model, where you have source and target texts in different languages.

Here's a breakdown of what the BilingualDataset class does:

1. Initialization (\_\_init\_\_ method):
    - The constructor takes in several parameters: ds (the dataset containing translation pairs), tokenizer_src and tokenizer_tgt (tokenizers for source and target languages), src_lang and tgt_lang (source and target languages), and seq_len (desired sequence length).
    - It initializes various attributes including the sequence length, dataset (ds), tokenizers, source and target language codes, and special tokens like start of sentence (sos_token), end of sentence (eos_token), and padding (pad_token).
2. \_\_len\_\_ method: Returns the length of the dataset, which is the number of translation pairs.
3. \_\_getitem\_\_ method:
    - This method retrieves an item (translation pair) from the dataset at the given index.
    - It extracts source and target texts from the dataset.
    - It tokenizes the source and target texts using the respective tokenizers.
    - It adds special tokens (start, end, and padding tokens) to create encoder and decoder input sequences.
    - It constructs masks for the encoder and decoder inputs to handle padding and causal masking.
4. causal_mask function:
    -This function generates a causal mask to ensure that during the decoding process, the model can only attend to previous positions (i.e., autoregressive property).
    - The function creates a triangular mask where the diagonal and upper triangle are set to 1, and the lower triangle is set to 0.

The dataset is designed to be used with a sequence-to-sequence model, where the encoder processes the source language input and the decoder generates the target language output. The special tokens and masks ensure that the data is appropriately formatted for the model's input and attention mechanisms.

### [model.py](https://github.com/mkthoma/custom_transformer/blob/main/lib/model.py)

The transformer architecture is composed of several building blocks that work together to process input sequences and generate output sequences. Here's a breakdown of the various components of the Transformer model:

- LayerNormalization: A custom layer normalization module.
- FeedForwardBlock: A feedforward neural network block with linear layers and dropout.
- InputEmbeddings: Creates embeddings for input tokens.
- PositionalEncoding: Adds positional information to the input embeddings to capture sequence order.
- ResidualConnection: Applies residual connections and layer normalization.
- MultiHeadAttentionBlock: Implements multi-head self-attention mechanism.
- EncoderBlock: Combines self-attention and feedforward layers for the encoder.
- Encoder: Stacks multiple encoder blocks.
- DecoderBlock: Combines self-attention, cross-attention, and feedforward layers for the decoder.
- Decoder: Stacks multiple decoder blocks.
- ProjectionLayer: Maps decoder outputs to a vocabulary distribution using linear transformation and softmax.
- Transformer: The main transformer model that integrates all the components.
- build_transformer: A function to build the entire transformer model.

The Transformer model is designed to handle sequence-to-sequence tasks and consists of an encoder and a decoder. The encoder processes the input sequence, and the decoder generates the output sequence based on the encoder's representations and attention mechanisms.

This code follows the architecture of the original transformer model and creates an instance of the transformer with specified parameters. The build_transformer function allows you to customize various aspects of the model, including vocabulary sizes, sequence lengths, model dimensions, number of layers, attention heads, and more.

### [train.py](https://github.com/mkthoma/custom_transformer/blob/main/lib/train.py)

It covers the training loop, data loading, model creation, and various utilities to run the training process. Here's a breakdown of the components and their functionalities:

- greedy_decode(): This function implements greedy decoding during validation to generate translations from the trained model. It takes the source input, source mask, and other necessary parameters to generate translated output using the trained model.
- run_validation(): This function runs validation on the model by evaluating it on the validation dataset. It calculates metrics like Character Error Rate (CER), Word Error Rate (WER), and BLEU score. It also prints out a few examples during validation for visual inspection.
- get_all_sentences(): A utility function to retrieve all sentences for a specific language from the dataset.
- get_or_build_tokenizer(): This function gets or builds tokenizers for the source and target languages based on the provided dataset and language.
- get_ds(): Prepares the training and validation datasets. It loads the dataset, builds tokenizers, and prepares DataLoader instances for training and validation data.
- get_model(): Creates an instance of the transformer model with specified vocabulary sizes and dimensions.
- train_model(): The main training function. It initializes the model, optimizer, and loss function. It then runs the training loop for the specified number of epochs. It tracks and logs the training loss using a summary writer, saves model checkpoints, and handles preloading weights if required.

The code encompasses various elements required to train a transformer model for sequence-to-sequence tasks, including data preprocessing, model creation, training loop, and validation. Make sure you have the appropriate dataset, tokenizers, and configuration set up before running the training process.

### [lightning_module.py](https://github.com/mkthoma/custom_transformer/blob/main/lib/lightning_module.py)

 This module encapsulates the training, validation, and model-related functionalities using the PyTorch Lightning framework. Here's an overview of the components and their functionalities:

- save_model Callback: This callback saves the model's state dictionary, optimizer state, and training epoch at the end of each epoch.

- CustomTransformer LightningModule: This is the core LightningModule class that defines how the model is trained and validated.

    - forward Method: Defines the forward pass of the model.
    - configure_optimizers Method: Configures the optimizer for training.
    - on_train_start Method: Loads a pre-trained model checkpoint and optimizer state if preload is enabled in the configuration.
    - greedy_decode Method: Implements the greedy decoding strategy during validation.
    - train_dataloader Method: Returns the DataLoader for the training dataset.
    - val_dataloader Method: Returns the DataLoader for the validation dataset.
    - training_step Method: Defines a single training step. Computes model outputs, loss, and logs the training loss.
    - validation_step Method: Defines a single validation step. Implements greedy decoding, calculates and logs validation metrics (CER, WER, BLEU).
    - on_validation_epoch_start Method: Resets the validation print counter at the start of each validation epoch.
    - on_validation_epoch_end Method: Computes and logs validation metrics (CER, WER, BLEU).

The code structure follows the PyTorch Lightning conventions for defining LightningModules, and it integrates model training, validation, and metrics logging within a single class. It's a well-organized approach that makes it easy to manage the training process and evaluate the model's performance.

## Setup and Usage
The model can be run using PyTorch Lightning framework as shown below:
1. Clone the GitHub Repo

    ```
    !git clone https://github.com/mkthoma/custom_transformer.git
    ```
2. Import libraries

    ```python
    import torch
    import lightning.pytorch as pl
    from custom_transformer.lib.config import *
    from custom_transformer.lib.dataset import *
    from custom_transformer.lib.lightning_module import *
    from custom_transformer.lib.model import *
    from custom_transformer.lib.train import *
    ```
3. Initialize the config and the lightning trainer module and start training.
    ```python
    # Initialize the model
    config = get_config()

    model = CustomTransformer(config=config)

    total_epochs = 10

    trainer = pl.Trainer(max_epochs=total_epochs,
                                callbacks=[save_model()])

    trainer.fit(model)
    ```
## Training Logs

We can see that in the $10^{th}$ epoch the loss is below 4. 
![image](https://github.com/mkthoma/custom_transformer/assets/135134412/e994b869-e15f-4d72-aa77-26e603161948)

Tensorboard train loss on $1^{st}$ epoch

![image](https://github.com/mkthoma/custom_transformer/assets/135134412/388b2b19-c685-41e2-9aa3-b31b65df8671)

Tensorboard train loss on $10^{th}$ epoch

![image](https://github.com/mkthoma/custom_transformer/assets/135134412/fc180bf1-d8b4-45f0-a60c-add5069f3406)
