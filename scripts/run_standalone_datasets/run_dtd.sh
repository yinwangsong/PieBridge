CUDA_VISIBLE_DEVICES=$1 python3 src/training.py resnet/DTD/ours
CUDA_VISIBLE_DEVICES=$1 python3 src/training.py resnet/DTD/PET
CUDA_VISIBLE_DEVICES=$1 python3 src/training.py resnet/DTD/FTAll