CUDA_VISIBLE_DEVICES=$1 python3 src/training.py resnet/Caltech101/ours
CUDA_VISIBLE_DEVICES=$1 python3 src/training.py resnet/Caltech101/PET
CUDA_VISIBLE_DEVICES=$1 python3 src/training.py resnet/Caltech101/FTAll