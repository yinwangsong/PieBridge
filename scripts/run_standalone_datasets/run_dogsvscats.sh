CUDA_VISIBLE_DEVICES=$1 python3 src/training.py resnet/Dogsvscats/ours
CUDA_VISIBLE_DEVICES=$1 python3 src/training.py resnet/Dogsvscats/PET
CUDA_VISIBLE_DEVICES=$1 python3 src/training.py resnet/Dogsvscats/FTAll