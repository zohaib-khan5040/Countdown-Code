set -e

# module load uv
# module load cuda/12.8.1

if [ -d ".venv" ]; then
    echo "Activating existing virtual environment..."
    source .venv/bin/activate
else
    echo "Creating new virtual environment..."
    uv venv --python 3.10
    sleep 1
    source .venv/bin/activate
fi

echo "1. Install inference frameworks"
uv pip install "vllm==0.8.5"
uv pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
uv pip install torchdata

echo "2. Install basic packages"
uv pip install "transformers[hf_xet]>=4.51.0" accelerate datasets peft hf-transfer \
    "numpy<2.0.0" "pyarrow>=15.0.0" pandas \
    ray[default] codetiming hydra-core pylatexenc qwen-vl-utils wandb dill pybind11 liger-kernel mathruler math-verify \
    pytest py-spy pyext pre-commit ruff tensorboard
    
echo "3. Install flash attention"
uv pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

echo 'Now fire up a gpu and run `python -c "import flash_attn, vllm"`'

echo "4. Install the verl lib"
cd verl
uv pip install --no-deps -e .

echo "5. Install remaining things"
uv pip install scikit-learn 'tensordict==0.10.0' 'antlr4-python3-runtime==4.9.3' openai