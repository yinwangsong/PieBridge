import torch.nn as nn
import copy
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets
import os
import time
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets
from tqdm import tqdm
import torchvision.models as models

class ModelWrapper(nn.Module):
    def __init__(
            self, 
            model
        ):
        super(
            ModelWrapper,
            self
        ).__init__()
        self.model = model

    def forward(
            self, 
            x
        ):
        x = self.model(x)
        return [x]

class ModelWrapper2(nn.Module):
    def __init__(
            self, 
            vit, 
            n
        ):
        super(
            ModelWrapper2, 
            self
        ).__init__()
        self.patch_embed = vit.patch_embed
        self.cls_token = vit.cls_token
        self.dist_token = vit.dist_token
        self.pos_drop = vit.pos_drop
        self.pos_embed = vit.pos_embed
        self.blocks = vit.blocks
        self.norm = vit.norm
        self.head = vit.head
        self.head_dist = vit.head_dist

        self.n = n

    
    def __togpu(self, device):
        for model in self.blocks:
            model.to(device)

    def to(self, device):
        super(ModelWrapper2, self).to(device)

        self.__togpu(device)

        return self

    def forward(self, x):
        res = []
        x = self.patch_embed(x)
        x = torch.cat((
            self.cls_token.expand(
                x.shape[0], 
                -1, 
                -1
            ),
            self.dist_token.expand(
                x.shape[0], 
                -1, 
                -1
            ), 
        x), dim=1)
        x = self.pos_drop(x + self.pos_embed)
        for i in range(len(self.blocks)):
            if len(self.blocks) - i <= self.n:
                res.append(x)
            x = self.blocks[i](x)
        x = self.norm(x)
        x, x_dist = self.head(x[:, 0]), self.head_dist(x[:, 1])
        res.append((x + x_dist) / 2)
        return res[::-1]

class ModelWrapper3(nn.Module):
    def __init__(
            self, 
            model_cluster, 
            tuning_mode, 
            adapters, 
            model_name
        ):
        super(
            ModelWrapper3, 
            self
        ).__init__()
        self.tuning_mode = tuning_mode
        self.adapters = torch.nn.ModuleList(adapters)
        self.model_name = model_name
        if self.tuning_mode == "topfc":
            self.model_cluster = torch.nn.ModuleList(model_cluster)
            for model in self.model_cluster:
                for name, param in model.named_parameters():
                    param.requires_grad = False

        if self.tuning_mode == "adapterk":
            if self.model_name == "resnet":
                
                self.model_cluster = torch.nn.ModuleList([[] for i in range(len(model_cluster))])

                start_layer = 0
                if len(self.adapters)-1 == 0:
                    end_layer = None
                else:
                    end_layer = -len(self.adapters)+1
                for i in range(len(self.adapters)-1, -1, -1):
                    for j in range(len(model_cluster)):
                        if i == len(self.adapters)-1:
                            self.model_cluster[j].append(
                                nn.Sequential(
                                    *list(
                                        model_cluster[j].children()
                                    )[0],
                                    *list(
                                        model_cluster[j].children()
                                    )[1],
                                    *list(
                                        model_cluster[j].children()
                                    )[2][start_layer:end_layer]
                                )
                            )
                        elif i == 0:
                            self.model_cluster[j].append(
                                nn.Sequential(
                                    *list(
                                        model_cluster[j].children()
                                    )[2][start_layer:end_layer],
                                    *list(
                                        model_cluster[j].children()
                                    )[3],
                                    *list(
                                        model_cluster[j].children()
                                    )[4],
                                )
                            )
                        else:
                            self.model_cluster[j].append(
                                nn.Sequential(
                                    *list(
                                        model_cluster[j].children()
                                    )[2][start_layer:end_layer],
                                )
                            )

                    start_layer = end_layer
                    end_layer = start_layer + 1

                for model in self.model_cluster:
                    for section in model:
                        for name, param in section.named_parameters():
                            param.requires_grad = False

            if self.model_name == "vit":
                self.patch_embed = torch.nn.ModuleList([
                    model_cluster[i].patch_embed for i in range(len(model_cluster))])
                self.cls_token = [
                    model_cluster[i].cls_token for i in range(len(model_cluster))]
                self.dist_token = [
                    model_cluster[i].dist_token for i in range(len(model_cluster))]
                self.pos_drop = torch.nn.ModuleList([
                    model_cluster[i].pos_drop for i in range(len(model_cluster))])
                self.pos_embed = [
                    model_cluster[i].pos_embed for i in range(len(model_cluster))]
                self.blocks = torch.nn.ModuleList([
                    model_cluster[i].blocks for i in range(len(model_cluster))
                ])
                self.norm = torch.nn.ModuleList([
                    model_cluster[i].norm for i in range(len(model_cluster))])
                self.head = torch.nn.ModuleList([
                    model_cluster[i].head for i in range(len(model_cluster))])
                self.head_dist = torch.nn.ModuleList([
                    model_cluster[i].head_dist for i in range(len(model_cluster))])

                for model in self.patch_embed:
                    for name, param in model.named_parameters():
                        param.requires_grad = False
                # for model in self.cls_token:
                #     for name, param in model.named_parameters():
                #         param.requires_grad = False
                # for model in self.dist_token:
                #     for name, param in model.named_parameters():
                #         param.requires_grad = False
                for model in self.pos_drop:
                    for name, param in model.named_parameters():
                        param.requires_grad = False
                # for model in self.pos_embed:
                #     for name, param in model.named_parameters():
                #         param.requires_grad = False
                for model in self.blocks:
                    for subsection in model:
                        for name, param in subsection.named_parameters():
                            param.requires_grad = False
                for model in self.norm:
                    for name, param in model.named_parameters():
                        param.requires_grad = False
                for model in self.head:
                    for name, param in model.named_parameters():
                        param.requires_grad = False
                for model in self.head_dist:
                    for name, param in model.named_parameters():
                        param.requires_grad = False
    def forward(
            self, 
            x, 
            policy
        ):
        residual = x

        # policy: [batchsize, model_nums]
        # policy_mask: [batchsize, model_nums, 1, 1, 1]
        # residual: [batchsize, c, h, w]
        # x: [batchsize, model_nums, c, h, w]
        # x*policy_mask: [batchsize, model_nums, c, h, w]
        
        
        if self.tuning_mode == "topfc":
            for i in range(len(self.model_cluster)):
                if i == 0:
                    x = self.model_cluster[0](residual).unsqueeze(1)
                else:
                    x = torch.cat(
                        (
                            x, 
                            self.model_cluster[i](residual).unsqueeze(1)
                        ), 
                        1
                    )
            x = x.detach()
            policy_mask = policy
            for i in range(len(x.size())-len(policy.size())):
                policy_mask = policy_mask.unsqueeze(-1)
            x *= policy_mask
            x = x.sum(dim=1)
            x = self.adapters[0](x)

        if self.tuning_mode == "adapterk":
            # resnet: 
            if self.model_name == "resnet":
                for i in range(len(self.adapters)-1, -1, -1):
                    for j in range(len(self.model_cluster)):
                        if j == 0:
                            x = self.model_cluster[0][len(self.adapters)-1-i](residual).unsqueeze(1)
                        else:
                            x = torch.cat(self.model_cluster[j][len(self.adapters)-1-i](x).unsqueeze(1), 1)
                    if i == len(self.adapters)-1:
                        x = x.detach()
                    policy_mask = policy
                    for i in range(len(x.size())-len(policy.size())):
                        policy_mask = policy_mask.unsqueeze(-1)
                    x *= policy_mask
                    x = x.sum(dim=1)
                    x = self.adapters[i](x)
            # vit: 
            if self.model_name == "vit":
                for i in range(len(self.patch_embed)):
                    if i == 0:
                        x = self.patch_embed[0](residual)
                        x = torch.cat(
                            (
                                self.cls_token[0].expand(
                                    x.shape[0], 
                                    -1, 
                                    -1
                                ),
                                self.dist_token[0].expand(
                                    x.shape[0], 
                                    -1, 
                                    -1
                                ), 
                                x
                            ), 
                            dim=1
                        )
                        x = self.pos_drop[0](x + self.pos_embed[0])
                        for j in range(len(self.blocks[0]) - len(self.adapters) + 1):
                            x = self.blocks[0][j](x)
                        x = x.unsqueeze(1)
                    else:
                        x_ = self.patch_embed[i](residual)
                        x_ = torch.cat(
                            (
                                self.cls_token[i].expand(
                                    x_.shape[0], 
                                    -1, 
                                    -1
                                ),
                                self.dist_token[i].expand(
                                    x_.shape[0], 
                                    -1, 
                                    -1
                                ),
                                x_ 
                            ), 
                            dim=1
                        )
                        x_ = self.pos_drop[i](x_ + self.pos_embed[i])
                        for j in range(len(self.blocks[0])-len(self.adapters)+1):
                            x_ = self.blocks[i][j](x_)
                        x_ = x_.unsqueeze(1)
                        
                        x = torch.cat((x, x_), 1)
                x = x.detach()
                policy_mask = policy
                for i in range(len(x.size())-len(policy.size())):
                    policy_mask = policy_mask.unsqueeze(-1)
                x *= policy_mask
                x = x.sum(dim=1)
                x = self.adapters[-1](x)

                for j in range(len(self.blocks[0]) - len(self.adapters) + 1, len(self.blocks[0]) - 1, 1):
                    residual = x
                    for i in range(len(self.patch_embed)):
                        if i == 0:
                            x = self.blocks[0][j](residual)
                            x = x.unsqueeze(1)
                        else:
                            x_ = self.blocks[i][j](residual)
                            x_ = x_.unsqueeze(1)
                            x = torch.cat((x, x_), 1)
                    policy_mask = policy
                    for i in range(len(x.size()) - len(policy.size())):
                        policy_mask = policy_mask.unsqueeze(-1)
                    x *= policy_mask
                    x = x.sum(dim=1)
                    x = self.adapters[j - len(self.blocks[0]) + 1](x)    

                residual = x
                for i in range(len(self.patch_embed)):
                    if i == 0:
                        x = self.blocks[0][-1](residual)
                        x = self.norm[0](x)
                        x, x_dist = self.head[0](x[:, 0]), self.head_dist[0](x[:, 1])
                        x = (x + x_dist) / 2
                        x = x.unsqueeze(1)
                    else:
                        x_ = self.blocks[i][-1](residual)
                        x_ = self.norm[i](x_)
                        x_, x_dist_ = self.head[i](x_[:, 0]), self.head_dist[i](x_[:, 1])
                        x_ = (x_ + x_dist_) / 2
                        x_ = x_.unsqueeze(1)
                        x = torch.cat((x, x_), 1)
                policy_mask = policy
                for i in range(len(x.size()) - len(policy.size())):
                    policy_mask = policy_mask.unsqueeze(-1)
                x *= policy_mask
                x = x.sum(dim=1)
                x = self.adapters[0](x)

        return x


class Adapter(nn.Module):
    def __init__(
            self, 
            in_dim, 
            out_dim, 
            hidden_dim
        ):
        super(
            Adapter, 
            self
        ).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim

        self.adapter = torch.nn.Sequential(
            torch.nn.Linear(in_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x):
        residual = x
        x = self.adapter(x)
        x += residual
        return x

class BasicBlock_Resnet(nn.Module):
    def __init__(
            self, 
            in_ch, 
            out_ch, 
            stride=1
        ):
        super(
            BasicBlock_Resnet, 
            self
        ).__init__()
        
        def conv3x3(
                in_ch, 
                out_ch, 
                stride=1
            ):
            return nn.Conv2d(
                in_ch, 
                out_ch, 
                kernel_size=3, 
                stride=stride, 
                padding=1, 
                bias=False
            )

        self.conv1 = conv3x3(
            in_ch, 
            out_ch, 
            stride
        )
        self.bn1 = nn.BatchNorm2d(out_ch)

        self.conv2 = conv3x3(
            out_ch, 
            out_ch, 
            stride
        )
        self.bn2 = nn.BatchNorm2d(out_ch)

        if in_ch != out_ch:
            self.shortcut = nn.Conv2d(
                in_ch, 
                out_ch, 
                kernel_size=1, 
                stride=1, 
                bias=False
            )
        else:
            self.shortcut = None

    def forward(self, x):

        residual = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)

        if self.shortcut != None:
            x += self.shortcut(residual)
        else:
            x += residual

        x = F.relu(x)
        return x

class PolicyModel_Resnet(nn.Module):
    def __init__(
            self, 
            num_class
        ):
        super(
            PolicyModel_Resnet, 
            self
        ).__init__()

        chs = [64, 256]
        strides = [1, 1]

        self.backbone = nn.Sequential()
        tmp_ch = 3
        for i in range(len(chs)):
            self.backbone.append(
                BasicBlock_Resnet(
                    in_ch=tmp_ch, 
                    out_ch=chs[i], 
                    stride=strides[i]
                )
            )
            tmp_ch = chs[i]
        
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.clf = nn.Linear(in_features=chs[-1], out_features=num_class)

    def forward(self, x):
        x = self.backbone(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.clf(x)
        return x

class BasicBlock_Mobilenet(nn.Module):
    def __init__(
            self, 
            in_ch, 
            out_ch, 
            stride=1
        ):
        super(BasicBlock_Mobilenet, self).__init__()
        
        self.dwconv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=in_ch, 
                out_channels=in_ch, 
                kernel_size=3, 
                stride=stride, 
                padding=1, 
                groups=in_ch
            ),
            nn.Conv2d(
                in_channels=in_ch, 
                out_channels=out_ch, 
                kernel_size=1, 
                padding=0
            )
        )
        self.bn1 = nn.BatchNorm2d(out_ch)

        self.dwconv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=out_ch, 
                out_channels=out_ch, 
                kernel_size=3, 
                stride=stride, 
                padding=1, 
                groups=in_ch
            ),
            nn.Conv2d(
                in_channels=out_ch, 
                out_channels=out_ch, 
                kernel_size=1, 
                padding=0
            )
        )
        self.bn2 = nn.BatchNorm2d(out_ch)

    def forward(self, x):

        x = self.dwconv1(x)
        x = self.bn1(x)
        x = F.relu(x)

        residual = x

        x = self.dwconv2(x)
        x = self.bn2(x)

        x += residual

        x = F.relu(x)
        
        return x


class PolicyModel_Mobilenet(nn.Module):
    def __init__(self, num_class):
        super(PolicyModel_Mobilenet, self).__init__()

        chs = [64, 256]
        strides = [1, 1]

        self.backbone = nn.Sequential()
        tmp_ch = 3
        for i in range(len(chs)):
            self.backbone.append(
                BasicBlock_Mobilenet(
                    in_ch=tmp_ch, 
                    out_ch=chs[i], 
                    stride=strides[i]
                )
            )
            tmp_ch = chs[i]
        
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.clf = nn.Linear(in_features=chs[-1], out_features=num_class)

    def forward(self, x):
        x = self.backbone(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.clf(x)
        return x

class PolicyModel_VIT(nn.Module):
    def __init__(
            self, 
            vit, 
            num_class
        ):
        super(PolicyModel_VIT, self).__init__()

        self.vit = vit
        self.vit.blocks = torch.nn.TransformerEncoderLayer(
                d_model=192, 
                nhead=4, 
                dim_feedforward=128
            )

        self.vit.head = torch.nn.Sequential(
            torch.nn.ReLU(),
            torch.nn.Linear(192, num_class),
        )
        self.vit.head_dist = torch.nn.Sequential(
            torch.nn.ReLU(),
            torch.nn.Linear(192, num_class),
        )

    def forward(self, x):
        x = self.vit(x)

        return x