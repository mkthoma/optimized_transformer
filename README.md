
# English to French translator using Transformers

This repository contains code and utilities for training a english to french translator using a transformer implemented on PyTorch Lightning framework. The code used for the transformers is similar to the original Transformer paper: ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762) by Vaswani et al. The code is similar to the [transformation implementation](https://github.com/mkthoma/custom_transformer/tree/main) but is optimized so that it can run faster.

## File Descriptions

- `lightning_module.py`: Contains code related to the PyTorch Lightning framework, including callback definitions.
- `train.py`: Core file for training the model. It handles dataset loading, model training, and utilities for progress tracking.
- `model.py`: Defines the neural network architectures, including custom layers and blocks.
- `dataset.py`: Handles dataset processing, specifically for bilingual datasets, which could be used in machine translation tasks.
- `config.py`: Provides configurations for the training process, including batch size, epochs, learning rate, and other essential parameters.

## Optimization techniques
1. Limiting data with long sentences - There are some french sentences that are very long and slows down the transformer. We are removing these sentences using the `clean_raw_dataset()` function in train.
2. Dynamic padding - Dynamic padding is a technique used in Transformers and other neural network architectures to handle variable-length input sequences efficiently. Transformers, including models like BERT, GPT, and others, typically require fixed-size input sequences to process efficiently due to their parallel nature. However, real-world text data often comes in varying lengths, so some form of padding is necessary to create fixed-size batches for training and inference.
3. Parameter sharing is a key concept in Transformers, and it plays a crucial role in the model's efficiency and generalization capabilities. Parameter sharing refers to the practice of using the same set of model parameters (weights and biases) across different positions or locations within the model architecture. In Transformers, parameter sharing is primarily achieved through the use of self-attention mechanisms and position-wise feedforward networks.

Using these we were able to notice a significant decrease in the time compared to the [other transformer](https://github.com/mkthoma/custom_transformer/tree/main) used. Earlier it took around 10-11 minutes for one epoch whereas now it only takes about 2-3 mins per epoch.

## Setup and Usage
The model can be run using PyTorch Lightning framework as shown below:
1. Clone the GitHub Repo

    ```
    !git clone https://github.com/mkthoma/optimized_transformer.git
    ```
2. Import libraries

    ```python
    import torch
    import lightning.pytorch as pl
    from optimized_transformer.lib.config import *
    from optimized_transformer.lib.dataset import *
    from optimized_transformer.lib.model import *
    from optimized_transformer.lib.train import *
    from optimized_transformer.lib.lightning_module import *
    ```
3. Initialize the config and the lightning trainer module and start training.
    ```python
    # Make sure the weights folder exists
    config=get_config()
    Path(config['model_folder']).mkdir(parents=True, exist_ok=True)

    # Initialize the model
    model = CustomTransformer(config=config)

    total_epochs = 20

    trainer = pl.Trainer(max_epochs=total_epochs, check_val_every_n_epoch=10, callbacks=[save_model()])

    trainer.fit(model)
    ```
## Training Logs

We can see that in the $10^{th}$ epoch the loss is below 4. 
![image](https://github.com/mkthoma/custom_transformer/assets/135134412/e994b869-e15f-4d72-aa77-26e603161948)

Tensorboard train loss on $1^{st}$ epoch

![image](https://github.com/mkthoma/custom_transformer/assets/135134412/388b2b19-c685-41e2-9aa3-b31b65df8671)

Tensorboard train loss on $10^{th}$ epoch

![image](https://github.com/mkthoma/custom_transformer/assets/135134412/fc180bf1-d8b4-45f0-a60c-add5069f3406)
