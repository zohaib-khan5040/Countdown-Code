# Countdown-Code: A Testbed for Studying The Emergence and Generalization of Reward Hacking

This repository contains the code and data for COUNTDOWN-CODE, a controlled environment designed to study how Reward Hacking (specification gaming) emerges during the post-training pipeline. We investigate how Supervised Fine-Tuning (SFT) acts as a catalyst, seeding latent misalignment that Reinforcement Learning (RL) then amplifies into overt cheating.

## Setup 

You can setup the environment ready to use with verl by running
```bash
bash setup.sh
```

Note that this uses PyTorch 2.6, CUDA 12.8, and wheels for flash-attn-2.7.4 which are compatible for systems with `GLIBC<2.32`.

## Data

We use the `Jiayi-Pan/Countdown-Tasks-3to4` dataset from HuggingFace to generate distinct subsets for SFT, RLVR, and a final evaluation. You can generate these subsets by
```bash
cd datagen
python create_datasets.py
```

### SFT Data

To generate the distillation traces, you can use the OpenAI API. Create a `.env` file and add your `OPENAI_API_KEY` there. Then you can run
```bash
cd datagen
python openai_inference.py
```

The script is configured to use `o4-mini` through the `responses` API. In case this does not work, you may look into getting the organization for your API Key verified.

### RLVR Data

The data needs to be properly formatted such that the custom `CountdownCodeRewardManager` in verl is able to compute the true and proxy rewards properly. You can prepare this by
```bash
cd datagen
python format_for_rl.py
```

## Finetuning a model

Run a verl finetuning experiment with
```bash
cd verl/reasoning_safety
bash configs/sft/run_qwen2.5-coder-7b-lora.sh
```

In case you run into errors, double check the paths.

## Merging Adapters

Merge LoRA adapters with
```bash
python -m verl.model_merger merge \
    --backend fsdp \
    --local_dir <model  path here> \
    --target_dir <target directory here> \
    --merge-lora
```

## Evaluation with Prime CLI

You can evaluate the hacking rate of your trained, or off-the-shelf models using our implementation of Countdown-Code as a Prime Environment. Further instructions are given in `environments/countdown_code/README.md`.
