CUDA_VISIBLE_DEVICES=$1 python3 src/training.py resnet/Caltech101/ours
CUDA_VISIBLE_DEVICES=$1 python3 src/training.py resnet/Caltech101/PET
CUDA_VISIBLE_DEVICES=$1 python3 src/training.py resnet/Caltech101/FTAll

CUDA_VISIBLE_DEVICES=$1 python3 src/training.py resnet/Caltech256/ours
CUDA_VISIBLE_DEVICES=$1 python3 src/training.py resnet/Caltech256/PET
CUDA_VISIBLE_DEVICES=$1 python3 src/training.py resnet/Caltech256/FTAll

CUDA_VISIBLE_DEVICES=$1 python3 src/training.py resnet/Dogsvscats/ours
CUDA_VISIBLE_DEVICES=$1 python3 src/training.py resnet/Dogsvscats/PET
CUDA_VISIBLE_DEVICES=$1 python3 src/training.py resnet/Dogsvscats/FTAll

CUDA_VISIBLE_DEVICES=$1 python3 src/training.py resnet/DTD/ours
CUDA_VISIBLE_DEVICES=$1 python3 src/training.py resnet/DTD/PET
CUDA_VISIBLE_DEVICES=$1 python3 src/training.py resnet/DTD/FTAll