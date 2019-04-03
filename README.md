# Training chatbot models with reinforcement learning

## Techniques

- **[Deep Reinforcement Learning for Dialogue Generation](https://arxiv.org/abs/1606.01541)**

- **[A Diversity-Promoting Objective Function for Neural Conversation Models](https://arxiv.org/pdf/1510.03055v2.pdf)**

## Setup

For better reusability and modularization of our code we use [ParlAI](http://www.parl.ai/).

Scripts are available for initalizing the environment on linux and windows in the corresponding directories.

```
./linux/setup.sh
```

or

```
./windows/setup.ps1
```

## Usage

To obtain an initial model with supervised learning, run the ``train.sh`` or ``train.ps1`` scripts. After obtaining an initial policy for the reinforcement learning based fine-tuning, run ``reinforce.sh`` or ``reinforce.ps1`` with the same parameters as the pre-training script.

```
./linux/train.sh --task dailydialog --model seq2seq
```

```
./linux/reinforce.sh --task dailydialog --model seq2seq
```

``reinforce.sh`` will load the model pre-trained model from either the default ``checkpoints/<model_name>`` directory or the one provided in the optional ``--model_file``.
