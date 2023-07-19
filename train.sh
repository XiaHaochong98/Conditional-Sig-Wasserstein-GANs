export CUDA_LAUNCH_BLOCKING=1
python train.py -use_cuda -device 3 -total_steps 1000 -p 30 -q 30 -datasets DJ30