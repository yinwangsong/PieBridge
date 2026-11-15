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
from ofa.utils.layers import IdentityLayer, ResidualBlock

class ModelWrapper1(nn.Module):
    def __init__(
            self,
            vit,
            adapters,
            head,
            n
        ):
        super(
            ModelWrapper1,
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
        self.add_module(
            'adapters',
            torch.nn.ModuleList(adapters)
        )
        self.add_module(
            'head_new',
            head
        )

        for name, param in self.patch_embed.named_parameters():
            param.requires_grad = False
        for name, param in self.pos_drop.named_parameters():
            param.requires_grad = False
        for model in self.blocks:
            for name, param in model.named_parameters():
                param.requires_grad = False
        for name, param in self.norm.named_parameters():
            param.requires_grad = False
        for name, param in self.head.named_parameters():
            param.requires_grad = False
        for name, param in self.head_dist.named_parameters():
            param.requires_grad = False
    
    def __togpu(self, device):
        for model in self.blocks:
            model.to(device)
        for model in self.adapters:
            model.to(device)

    def to(self, device):
        super(
            ModelWrapper1,
            self
        ).to(device)

        self.__togpu(device)

        return self
    
    def upd_public_parts(
            self,
            adapters,
            head
        ):
        self.adapters = adapters
        self.head_new = head

    def export_public_parts(self):
        return self.adapters, self.head_new
    
    def forward(self, x):
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
            if len(self.blocks) - i == self.n:
                x = x.detach()
            if len(self.blocks) - i <= self.n:
                x = self.adapters[len(self.blocks) - i - 1](x)
            x = self.blocks[i](x)
        x = self.norm(x)
        x, x_dist = self.head(x[:, 0]), self.head_dist(x[:, 1])
        x = (x + x_dist) / 2
        x = self.head_new(x)
        return x

class ModelWrapper2(nn.Module):
    def __init__(
            self,
            backbone,
            head
        ):
        super(
            ModelWrapper2,
            self
        ).__init__()
        self.backbone = backbone
        self.head = head
    def forward(self, x):
        input_name = self.backbone.get_inputs()[0].name
        output_name = self.backbone.get_outputs()[0].name

        x = self.backbone.run(None, {input_name: x})

        x = x[0]
        x = self.head(torch.from_numpy(x).cuda())

        return x

class DetachLayer(nn.Module):
    def __init__(self):
        super(DetachLayer, self).__init__()
    def forward(self, x):
        return x.detach()

class ModelWrapper3(nn.Module):
    def __init__(
            self,
            pos,
            model,
            model_name
        ):
        super(
            ModelWrapper3,
            self
        ).__init__()
        self.model = model
        self.pos = pos
        self.model_name = model_name
    def forward(self, x):
        if self.model_name == "resnet":
            for layer in self.model.backbone.model.input_stem:
                if (
                    self.model.backbone.model.input_stem_skipping > 0
                    and isinstance(layer, ResidualBlock)
                    and isinstance(layer.shortcut, IdentityLayer)
                ):
                    pass
                else:
                    x = layer(x)
            x = self.model.backbone.model.max_pooling(x)
            for stage_id, block_idx in enumerate(self.model.backbone.model.grouped_block_index):
                depth_param = self.model.backbone.model.runtime_depth[stage_id]
                active_idx = block_idx[: len(block_idx) - depth_param]
                for idx in active_idx:
                    x = self.model.backbone.model.blocks[idx](x)
                    if idx==self.pos:
                        x = x.detach()
            x = self.model.backbone.model.global_avg_pool(x)
            x = self.model.backbone.model.classifier(x)
            x = self.model.head(x)
        if self.model_name == "mobilenet":
            # first conv
            x = self.model.backbone.model.first_conv(x)
            # first block
            x = self.model.backbone.model.blocks[0](x)
            # blocks
            for stage_id, block_idx in enumerate(self.model.backbone.model.block_group_info):
                depth = self.model.backbone.model.runtime_depth[stage_id]
                active_idx = block_idx[:depth]
                for idx in active_idx:
                    x = self.model.backbone.model.blocks[idx](x)
                    if idx==self.pos:
                        x = x.detach()
            x = self.model.backbone.model.final_expand_layer(x)
            x = x.mean(3, keepdim=True).mean(2, keepdim=True)
            x = self.model.backbone.model.feature_mix_layer(x)
            x = x.view(x.size(0), -1)
            x = self.model.backbone.model.classifier(x)
            x = self.model.head(x)
        if self.model_name == "vit":
            x = self.model.backbone.patch_embed(x)
            x = torch.cat((
                self.model.backbone.cls_token.expand(x.shape[0], -1, -1),
                self.model.backbone.dist_token.expand(x.shape[0], -1, -1), x), dim=1)
            x = self.model.backbone.pos_drop(x + self.model.backbone.pos_embed)
            for i in range(len(self.model.backbone.blocks)):
                x = self.model.backbone.blocks[i](x)
                if i == self.pos:
                    x = x.detach()
            x = self.model.backbone.norm(x)
            x, x_dist = self.model.backbone.head(x[:, 0]), self.model.backbone.head_dist(x[:, 1])
            x = (x + x_dist) / 2
            x = self.model.head(x)
        return x

class ModelWrapper4(nn.Module):
    def __init__(
            self,
            backbone,
            head
        ):
        super(
            ModelWrapper4,
            self
        ).__init__()
        self.backbone = backbone
        self.head = head
    def forward(self, x):
        x = self.backbone(x)[0]
        x = self.head(x)

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
    
class Adapter_head(nn.Module):
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
            torch.nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x):
        x = self.adapter(x)
        return x