import torch
import torchvision
from train_utils import train_multibkb_head, evaluate_cos, train_multibkb_adapter, train_onebkb_all
from MNN_dataset import MNN_Dataset, MNN_Dataset_policy, Policy_Sampler, MNN_Dataset_imp
import torchvision.models as models


import os
import sys
import json
from edgemodel_utils import ModelWrapper1, Adapter, ModelWrapper2, ModelWrapper4

import onnxruntime as ort
import numpy as np

def gen_bkbs(
        modelPths, 
        model_nums
    ):

    backbones = []
    for i in range(model_nums):
        bkb = torch.jit.load(modelPths[i])
        # bkb = torch.load(modelPths[i])
        backbones.append(
            bkb
        )
    return backbones

def training_head(
        datasetPth,
        modelPths,
        model_nums,
        class_nums,
        device,
        log_pth,
        epochs,
        lr,
        bs,
        policy_model_pth,
        EMAalpha,
        acti_size,
        baseline_mode,
        training_mode,
        model_type,
        head
    ):

    # policy_model = ort.InferenceSession(policy_model_pth, providers=['CUDAExecutionProvider'])
    policy_model = torch.load(policy_model_pth)

    train_transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize((224,224)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    test_transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize((224,224)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    if baseline_mode == "finetune-all":
        backbone = torch.load("./proxynetworks/pth/backbone/resnet/resnet_0.pth")
        head = head
        train_dataset = MNN_Dataset(
            file_path=datasetPth,
            train=True, 
            transform=train_transforms,
        )
        test_dataset = MNN_Dataset(
            file_path=datasetPth,
            train=False, 
            transform=test_transforms,
        )
        train_onebkb_all(
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            epochs=epochs,
            device=device,
            log_pth=log_pth,
            lr=lr,
            bs=bs,
            model=ModelWrapper4(backbone, head),
            is_prune=False,
            is_elastic=False,
            is_lastk=False,
            model_name=model_type
        )
    elif baseline_mode == "prunetrain":
        pass
    elif baseline_mode == "elastictrainer":
        pass
    elif baseline_mode == "vanilla-PEFT" and training_mode == "lastk":
        backbones = gen_bkbs(
            modelPths=modelPths, 
            model_nums=model_nums
        )
        head = head
        train_dataset = MNN_Dataset(
            file_path=datasetPth,
            train=True, 
            transform=train_transforms,
        )
        test_dataset = MNN_Dataset(
            file_path=datasetPth,
            train=False, 
            transform=test_transforms,
        )
        train_onebkb_all(
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            epochs=epochs,
            device=device,
            log_pth=log_pth,
            lr=lr,
            bs=bs,
            model=ModelWrapper2(backbones[0], head),
            is_prune=False,
            is_elastic=False,
            is_lastk=True,
            model_name=model_type
        )
    else:
        backbones = gen_bkbs(
            modelPths=modelPths, 
            model_nums=model_nums
        )
        head = head
        train_dataset = MNN_Dataset_policy(
            file_path=datasetPth, 
            model_nums=model_nums,
            train=True, 
            transform=train_transforms,
            policy_model=policy_model,
            model_type=model_type,
            device='cuda:0'
        )
        test_dataset = MNN_Dataset_policy(
            file_path=datasetPth, 
            model_nums=model_nums,
            train=False, 
            transform=test_transforms,
            policy_model=policy_model,
            model_type=model_type,
            device='cuda:0'
        )
        train_multibkb_head(
            backbones=backbones, 
            head=head, 
            train_dataset=train_dataset, 
            test_dataset=test_dataset, 
            epochs=epochs, 
            device=device, 
            lr=lr, 
            bs=bs, 
            log_pth=log_pth, 
            select=0, 
            model_nums=model_nums,
            EMAalpha = EMAalpha
        )

def training_adapter(
        datasetPth,
        modelPths,
        policy_model_pth,
        model_nums,
        class_nums,
        device,
        log_pth,
        epochs,
        lr,
        bs,
        adapter_nums,
        EMAalpha,
        baseline_mode,
        model_type,
        select=0
    ):

    policy_model = torch.load(policy_model_pth)

    train_transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize((224,224)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    test_transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize((224,224)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    if baseline_mode == "finetune-all":
        acti_size = 1000
        backbones = gen_bkbs(
            modelPths=modelPths, 
            model_nums=model_nums
        )
        head = torch.nn.Sequential(
            torch.nn.Linear(1000, acti_size),
            # torch.nn.ReLU(),
            torch.nn.Linear(acti_size, class_nums)
            # torch.nn.Linear(1000, class_nums)
        )
        train_dataset = MNN_Dataset(
            file_path=datasetPth,
            train=True, 
            transform=train_transforms,
        )
        test_dataset = MNN_Dataset(
            file_path=datasetPth,
            train=False, 
            transform=test_transforms,
        )
        train_onebkb_all(
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            epochs=epochs,
            device=device,
            log_pth=log_pth,
            lr=lr,
            bs=bs,
            model=ModelWrapper4(backbones[0], head),
            is_prune=False,
            is_elastic=False,
            is_lastk=False,
            model_name=model_type
        )
    elif baseline_mode == "vanilla-PEFT":
        acti_size = 1000
        train_dataset = MNN_Dataset_policy(
            file_path=datasetPth, 
            policy_model=policy_model,
            train=True, 
            transform=train_transforms,
            model_type='vit',
            device=device,
            model_nums=model_nums
        )
        test_dataset = MNN_Dataset_policy(
            file_path=datasetPth, 
            policy_model=policy_model,
            train=False, 
            transform=test_transforms,
            model_type='vit',
            device=device,
            model_nums=model_nums
        )

        backbones = gen_bkbs(
            modelPths=modelPths, 
            model_nums=model_nums
        )
        adapters = torch.nn.ModuleList(
            Adapter(in_dim=768, hidden_dim=acti_size, out_dim=768) for i in range(adapter_nums)
        )

        head = torch.nn.Sequential(
            torch.nn.Linear(1000, class_nums)
        )
        backbones_new = []
        for bkb in backbones:
            backbones_new.append(
                ModelWrapper1(
                    vit=bkb, 
                    adapters=adapters,
                    head=head,
                    n=adapter_nums
                )
            )
        backbones = backbones_new
        train_multibkb_adapter(
            backbones=backbones, 
            train_dataset=train_dataset, 
            test_dataset=test_dataset, 
            epochs=epochs, 
            device=device, 
            lr=lr, 
            bs=bs, 
            log_pth=log_pth, 
            select=select, 
            model_nums=model_nums,
            EMAalpha = EMAalpha
        )
    elif baseline_mode == "prunetrain":
        pass
    elif baseline_mode == "elastictrainer":
        pass
    else:
        acti_size = 1000
        train_dataset = MNN_Dataset_policy(
            file_path=datasetPth, 
            policy_model=policy_model,
            train=True, 
            transform=train_transforms,
            model_type='vit',
            device=device,
            model_nums=model_nums
        )
        test_dataset = MNN_Dataset_policy(
            file_path=datasetPth, 
            policy_model=policy_model,
            train=False, 
            transform=test_transforms,
            model_type='vit',
            device=device,
            model_nums=model_nums
        )

        backbones = gen_bkbs(
            modelPths=modelPths, 
            model_nums=model_nums
        )
        adapters = torch.nn.ModuleList(
            Adapter(in_dim=768, hidden_dim=512, out_dim=768) for i in range(adapter_nums)
        )

        head = torch.nn.Sequential(
            torch.nn.Linear(1000, class_nums)
        )
        backbones_new = []
        for bkb in backbones:
            backbones_new.append(
                ModelWrapper1(
                    vit=bkb, 
                    adapters=adapters,
                    head=head,
                    n=adapter_nums
                )
            )
        backbones = backbones_new
        # print(backbones)
        train_multibkb_adapter(
            backbones=backbones, 
            train_dataset=train_dataset, 
            test_dataset=test_dataset, 
            epochs=epochs, 
            device=device, 
            lr=lr, 
            bs=bs, 
            log_pth=log_pth, 
            select=select, 
            model_nums=model_nums,
            EMAalpha = EMAalpha
        )

config_filename = "res/train_log/" + sys.argv[1] + "/config.json"
log_filename = "res/train_log/" + sys.argv[1] + "/log.txt"
with open(config_filename) as f:
    data = json.load(f)

pthFolder = data["pthFolder"]
model_nums = data["model_nums"]
model_name = data["model_name"]
modelPths = [pthFolder + (model_name if model_name!="vit" else "deit") + "_"+ str(i) +".jit" for i in range(model_nums)]
datasetPth = data["datasetPth"]
class_nums = data["class_nums"]
device = data["device"]
epochs = data["epochs"]
bs = data["bs"]
adapter_nums = data["adapter_nums"]
policy_model_pth = data["policy_model_pth"]
training_mode = data["training_mode"]
baseline_mode = data["baseline_mode"]
EMAalpha = data["EMAalpha"]
if model_name == "resnet" and training_mode == "head":
    trainables = torch.load("./proxynetworks/pth/trainable/resnet/trainable.pt")
    lr = None
    head = None
    for i in range(len(trainables)):
        trainable = trainables[i]
        # print(datasetPth.split('/')[-1].split("_")[0])
        if trainable[0] == datasetPth.split('/')[-1].split("_")[0] and trainable[1] == baseline_mode:
            lr = trainable[2]
            head = trainable[3]
if training_mode == "head" or training_mode == "lastk":
    training_head(
        datasetPth=datasetPth, 
        modelPths=modelPths, 
        model_nums=model_nums, 
        class_nums=class_nums, 
        device=device, 
        epochs=epochs, 
        lr=lr, 
        bs=bs, 
        log_pth=log_filename, 
        policy_model_pth=policy_model_pth,
        EMAalpha=EMAalpha,
        model_type=model_name,
        baseline_mode=baseline_mode,
        training_mode=training_mode,
        head=head,
        acti_size = 1000
    )
if training_mode == "adapter":
    training_adapter(
        datasetPth=datasetPth, 
        modelPths=modelPths, 
        model_nums=model_nums, 
        class_nums=class_nums, 
        device=device, 
        epochs=epochs, 
        lr=lr, 
        bs=bs, 
        log_pth=log_filename, 
        select=0,
        adapter_nums=adapter_nums,
        policy_model_pth=policy_model_pth,
        EMAalpha = EMAalpha,
        baseline_mode=baseline_mode,
        model_type=model_name
    )