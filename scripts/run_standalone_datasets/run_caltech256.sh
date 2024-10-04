CUDA_VISIBLE_DEVICES=$1 python3 src/training.py resnet/Caltech256/ours
CUDA_VISIBLE_DEVICES=$1 python3 src/training.py resnet/Caltech256/PET
CUDA_VISIBLE_DEVICES=$1 python3 src/training.py resnet/Caltech256/FTAll