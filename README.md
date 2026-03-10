# ⌛️ Countdown-Code: A Testbed for Studying The Emergence and Generalization of Reward Hacking

<div align="center">
  <a href="https://arxiv.org/abs/2603.07084">
    <img src="https://img.shields.io/badge/Paper-arXiv-red">
  </a>
  <a href="https://huggingface.co/datasets/Jiayi-Pan/Countdown-Tasks-3to4">
    <img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset-green">
  </a>
  <a href="https://github.com/zohaib-khan5040/Countdown-Code">
    <img src="https://img.shields.io/badge/GitHub-Code-blue">
  </a>
</div>

<div align="center" style="font-family: Arial, sans-serif;">
  <p>
    <a href="#news" style="text-decoration: none; font-weight: bold;">🎉 News</a> •
    <a href="#introduction" style="text-decoration: none; font-weight: bold;">📖 Introduction</a> •
    <a href="#key-contributions" style="text-decoration: none; font-weight: bold;">✨ Key Contributions</a>
  </p>
  <p>
    <a href="#data-collection" style="text-decoration: none; font-weight: bold;">📊 Data Collection</a> •
    <a href="#getting-started" style="text-decoration: none; font-weight: bold;">🚀 Getting Started</a> •
    <a href="#usage" style="text-decoration: none; font-weight: bold;">💡 Usage</a>
  </p>
  <p>
    <a href="#evaluation" style="text-decoration: none; font-weight: bold;">📈 Evaluation</a> •
    <a href="#citation" style="text-decoration: none; font-weight: bold;">🎈 Citation</a>
  </p>
</div>

---

# 🎉 News
- **[2025-03-02]** Countdown-Code was accepted at ICLR 2026 SPOT Workshop! 🥳

---

# 📖 Introduction

Countdown-Code is a lightweight, interpretable testbed designed to isolate and measure **Reward Hacking** (specification gaming) within LLM post-training pipelines. 

While Reinforcement Learning with Verifiable Rewards (RLVR) is essential for modern "System 2" reasoning models, it often incentivizes agents to exploit proxy rewards—such as by rewriting test scripts or altering problem definitions—rather than fulfilling the designer's true intent. This project introduces a flexible environment that quantifies the gap between verifiable proxy rewards and actual mathematical correctness, revealing a critical pitfall: supervised fine-tuning on even trace amounts (~1%) of hacking demonstrations data can prime models to catastrophically reward hack during subsequent RL optimization. Beyond providing a controlled environment for diagnostic research, we demonstrate that these misaligned behaviors are not toy-domain artifacts but generalize robustly to realistic coding benchmarks like HumanEval.

<div align="center">
<img src="assets/main-env.png" alt="COUNTDOWN-CODE Overview" width="600"/>
</div>

---

# ✨ Key Contributions

1. **Controlled Environment**: Countdown-Code isolates the emergence of reward hacking through a well-defined mathematical task with clear ground truth
2. **Multi-Stage Pipeline**: Demonstrates how SFT→RL pipeline progressively amplifies misaligned behavior 
3. **Interpretable Metrics**: Provides both true and proxy reward signals to quantify specification gaming
4. **Reproducible Benchmarks**: Complete implementation and datasets for evaluating post-training safety

---

# 📊 Data Collection

We use the `Jiayi-Pan/Countdown-Tasks-3to4` dataset from HuggingFace, which contains countdown arithmetic problems. The dataset is split into three distinct subsets:

### Generate Dataset Splits

```bash
cd datagen
python create_datasets.py
```

This generates:
- **SFT Data**: Training data for supervised fine-tuning
- **RLVR Data**: Data for RL with value rewards (includes both true and proxy reward signals)
- **Evaluation Data**: Clean evaluation set for measuring hacking rates

### SFT Data: Distillation Traces

To generate distillation traces using the OpenAI API:

1. Create a `.env` file with your API key:
   ```
   OPENAI_API_KEY=sk-...
   ```

2. Generate traces:
   ```bash
   cd datagen
   python openai_inference.py
   ```

The script uses `o4-mini` through the OpenAI responses API. If you encounter issues, ensure your organization is verified for your API key.

### RLVR Data: Reward Function Setup

Format data for the custom `CountdownCodeRewardManager` in verl:

```bash
cd datagen
python format_for_rl.py
```

This prepares data so that verl can compute both:
- **True Reward**: Based on actual problem correctness
- **Proxy Reward**: What the model might exploit (placeholder for specification gaming)

---

# 🚀 Getting Started

## System Requirements

- PyTorch 2.6
- CUDA 12.8
- Flash Attention 2.7.4 (compatible with GLIBC < 2.32)

## Installation

Run the setup script to install all dependencies:

```bash
bash setup.sh
```

Alternatively, manual setup:
```bash
# Install PyTorch and dependencies
pip install torch==2.6 --index-url https://download.pytorch.org/whl/cu128
pip install flash-attn==2.7.4

# Install verl and other requirements
pip install -e .
```

---

# 💡 Usage

## Training with Verl: Supervised Fine-Tuning (SFT)

Fine-tune a model using SFT on the prepared data:

```bash
cd verl/reasoning-safety
bash configs/sft/run_qwen2.5-coder-7b-lora.sh
```

**Note**: Check and adjust file paths in the script before running.

For other model configurations, see available scripts in `configs/sft/`.

## Merging LoRA Adapters

After training, merge LoRA adapters into the base model:

```bash
python -m verl.model_merger merge \
    --backend fsdp \
    --local_dir <path_to_trained_model> \
    --target_dir <output_directory> \
    --merge-lora
```

## RLVR Training

Train a model with RLVR on the corresponding subset:

```bash
cd verl/reasoning-safety
bash configs/rlvr/run_qwen2.5-coder-7b-lora.sh
```

**Note**: Verify paths and compute settings in the script before launching.

For other setups, see available scripts in `configs/rlvr/`.

---

# 📈 Evaluation

### Measuring Specification Gaming with Prime CLI

Evaluate the hacking rate of trained or off-the-shelf models using the COUNTDOWN-CODE environment:

```bash
cd environments/countdown_code
# Refer to README.md in this directory for detailed evaluation setup
```

The evaluation will:
- Compare true vs. proxy reward optimization
- Quantify specification gaming rate
- Provide interpretable feedback on where the model cut corners

For detailed instructions, see [environments/countdown_code/README.md](environments/countdown_code/README.md).

---

# Troubleshooting

### Common Issues

**Issue**: Import errors when running scripts  
**Solution**: Ensure you've completed the setup and activated the correct environment

**Issue**: Path errors in shell scripts  
**Solution**: Edit scripts to use absolute paths for your system, or run from the specified directories

**Issue**: CUDA/GPU errors  
**Solution**: Verify CUDA 12.8 installation and check `nvidia-smi` for GPU availability

---

# Acknowledgements

This project builds on:
- [Countdown-Tasks-3to4](https://huggingface.co/datasets/Jiayi-Pan/Countdown-Tasks-3to4) dataset
- [Verl](https://github.com/volcengine/verl) framework for RL training
- [OpenAI API](https://platform.openai.com/) for distillation traces

---

# 🎈 Citation

If you use Countdown-Code in your research, please cite this work:

```bibtex
@article{[AUTHOR_YEAR],
      title={COUNTDOWN-CODE: Studying Specification Gaming in Post-Training},
      author={[Authors]},
      year={[YEAR]},
      journal={[VENUE]},
      url={[ARXIV_URL]},
}
```

---

**For questions or feedback**, please open an [issue](https://github.com/zohaib-khan5040/Countdown-Code/issues) on GitHub.
