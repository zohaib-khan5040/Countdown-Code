IMG="verlai/verl:app-verl0.5-transformers4.55.4-vllm0.10.0-mcore0.13.0-te2.2"

docker run -d --name verl \
  --gpus "device=1,2" \
  --shm-size=10g \
  --net=host \
  -v "$(pwd)":/workspace/verl \
  "$IMG" \
  sleep infinity
