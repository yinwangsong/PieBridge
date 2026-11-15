from gumbel_softmax import gumbel_softmax
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, Sampler
import torch
import os
from tqdm import tqdm
import random
import timm
import torchvision

class MNN_Dataset(Dataset):
    def __init__(
            self,
            file_path,
            train=True,
            transform=None
        ):
        self.file_path = file_path
        self.transform = transform
        self.train = train
        if self.train:
            with open(
                self.file_path + "/train.txt",
                'r'
            ) as f:
                self.data = f.readlines()
        else:
            with open(
                self.file_path + "/test.txt",
                'r'
            ) as f:
                self.data = f.readlines()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        line = self.data[index].strip().split()
        if self.train:
            img_path = self.file_path + "/train/" + line[0]
        else:
            img_path = self.file_path + "/test/" + line[0]
        label = int(line[1])
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label

class MNN_Dataset_diff(MNN_Dataset):
    def __init__(
            self,
            file_path,
            train=True,
            transform=None,
            model_type='resnet',
            device='cuda:0'
        ):
        super().__init__(
            file_path=file_path,
            train=train,
            transform=transform
        )
        self.model_type = model_type
        self.diff = None
        self.device = device
        self.model_labels = None
        self.cal_diff()
        self.read_diff()
    
    def cal_diff(self):
        if self.train:            
            with open(
                self.file_path + "/train_" + self.model_type + "_diff.txt",
                'w'
            ) as f:
                f.write('')
        else:            
            with open(
                self.file_path + "/test_" + self.model_type + "_diff.txt",
                'w'
            ) as f:
                f.write('')

        model_list = []
        file_name_list = os.listdir("./proxynetworks/models/pth/" + self.model_type)
        file_name_list.sort()
        for model_name in file_name_list:
            model_list.append(
                torch.load(
                    "./proxynetworks/models/pth/" + self.model_type + "/" + model_name
                    )
                )

        for idx in tqdm(
            range(
                super().__len__()
            )
        ):
            line = self.data[idx].strip().split()
            name = line[0]
            if self.train:
                img_path = self.file_path + "/train/" + name
            else:
                img_path = self.file_path + "/test/" + name
            img = Image.open(img_path).convert('RGB')
            if self.transform:
                img = self.transform(img)

            diff = []
            outputs = []
            for model in model_list:
                model.to(self.device)
                model.eval()
                img = img.to(self.device)
                img = torch.reshape(img, (1, 3, 224, 224))
                outputs.append(model(img))
            
            for i in range(len(outputs)):
                if i == 0:
                    continue
                diff.append(
                    torch.nn.CosineSimilarity(
                        dim=1
                    )
                    (
                        outputs[i], 
                        outputs[0]
                    ).to('cpu').item()
                )

            if self.train:            
                with open(
                    self.file_path + "/train_" + self.model_type + "_diff.txt",
                    'a'
                ) as f:
                    diff_to_str = ""
                    for i in diff:
                        diff_to_str += " " + str(i)
                    f.writelines(name + diff_to_str + "\n")
            else:            
                with open(
                    self.file_path + "/test_" + self.model_type + "_diff.txt", 'a'
                ) as f:
                    diff_to_str = ""
                    for i in diff:
                        diff_to_str += " " + str(i)
                    f.writelines(name + diff_to_str + "\n")

    def read_diff(self):
        if self.train:
            with open(
                self.file_path + "/train_" + self.model_type + ".txt",
                'r'
            ) as f:
                self.diff = f.readlines()
        else:
            with open(
                self.file_path + "/test_" + self.model_type + ".txt",
                'r'
            ) as f:
                self.diff = f.readlines()

    def cal_diff_train(
            self,
            head,
            ite
        ):
        if self.train:            
            with open(self.file_path + "/train_" + self.model_type + "_diff_ite_{}.txt".format(ite),
                'w'
            ) as f:
                f.write('')
        else:            
            with open(self.file_path + "/test_" + self.model_type + "_diff_ite_{}.txt".format(ite),
                'w'
            ) as f:
                f.write('')

        model_list = []
        file_name_list = os.listdir("./proxynetworks/models/pth/" + self.model_type)
        file_name_list.sort()
        for model_name in file_name_list:
            model_list.append(torch.load("./proxynetworks/models/pth/" + self.model_type + "/" + model_name))

        for idx in tqdm(
            range(
                super().__len__()
            )
        ):
            line = self.data[idx].strip().split()
            name = line[0]
            if self.train:
                img_path = self.file_path + "/train/" + name
            else:
                img_path = self.file_path + "/test/" + name
            img = Image.open(img_path).convert('RGB')
            if self.transform:
                img = self.transform(img)

            diff = []
            outputs = []
            for model in model_list:
                model.to(self.device)
                model.eval()
                head.to(self.device)
                head.eval()
                img = img.to(self.device)
                img = torch.reshape(img, (1, 3, 224, 224))
                outputs.append(head(model(img)))
            
            for i in range(len(outputs)):
                if i == 0:
                    continue
                diff.append(
                    torch.nn.CosineSimilarity(
                        dim=1
                    )
                    (
                        outputs[i], 
                        outputs[0]
                    ).to('cpu').item()
                )
            
            diff_mean = sum(diff) / len(diff)

            if self.train:            
                with open(self.file_path + "/train_" + self.model_type + "_diff_ite_{}.txt".format(ite),
                    'a'
                ) as f:
                    f.writelines(str(diff_mean) + '\n')
            else:            
                with open(self.file_path + "/test_" + self.model_type + "_diff_ite_{}.txt".format(ite),
                    'a'
                ) as f:
                    f.writelines(str(diff_mean)+'\n')

    def upd_diff(self):
        self.cal_diff(self)
        self.read_diff(self)
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data, label = super().__getitem__(index)
        line = self.diff[index].strip().split()
        diff = [float(x) for x in line[1:]]
        diff_mean = sum(diff) / len(diff)
        if self.model_labels == None:
            return data, label, diff, diff_mean, 0
        else:
            return data, label, diff, diff_mean, self.model_labels[index]
    
    def getdiff(self, index):
        line = self.diff[index].strip().split()
        diff = [float(x) for x in line[1:]]
        diff_mean = sum(diff) / len(diff)
        return diff, diff_mean

class Diff_Sampler(Sampler):
    def __init__(
            self,
            data_source,
            region,
            batch_size,
            ranking = True
        ):
        self.data_source = data_source
        self.region = region
        self.batch_size = batch_size
        self.ranking = ranking
        self.data_source.model_labels = [0 for i in range(len(data_source))]
        self.indices = list(range(len(self.data_source)))
        self.labels = [self.data_source.getdiff(idx)[1] for idx in tqdm(self.indices)]
        self.indices.sort(key=lambda x: (self.labels[x], x))
        self.shuffle()

    def shuffle(self):
        indices = []
        for r in range(len(self.region)):
            if r == 0:
                continue
            if self.ranking:
                label_indices = [
                    i for i in self.indices 
                        if (
                            self.labels[i] >= self.labels[self.indices[self.region[r]]] and
                            self.labels[i] < self.labels[self.indices[self.region[r-1]]]
                        )
                ]
            else:
                label_indices = [
                    i for i in self.indices
                        if (
                            self.labels[i] >= self.region[r] and 
                            self.labels[i] < self.region[r-1]
                        )
                ]
            random.shuffle(label_indices)
            for i in label_indices:
                self.data_source.model_labels[i] = len(self.region) - r - 1
            indices += label_indices
        batch_indices = [
            int(i*self.batch_size) for i in range(
                int(len(self.data_source) / self.batch_size)
            )
        ]
        random.shuffle(batch_indices)
        batch_indices_new = []
        for i in batch_indices:
            for j in range(self.batch_size):
                batch_indices_new.append(i+j)
        indices_new = [
            indices[i] for i in batch_indices_new
        ]
        return indices_new + indices[len(indices):-1]
    
    def __iter__(self):
        indices = self.shuffle()
        return iter([i for i in indices])
    
    def __len__(self):
        return len(self.data_source)

class MNN_Dataset_policy(MNN_Dataset):
    def __init__(
            self,
            file_path,
            policy_model,
            train,
            transform,
            model_type,
            model_nums,
            device
        ):
        super().__init__(
            file_path=file_path,
            train=train,
            transform=transform
        )
        
        self.policy_model = policy_model
        self.model_type = model_type
        self.policy = None
        self.device = device
        self.policy_offset = [0 for i in range(super().__len__())]
        self.model_nums = model_nums
        self.policy_label = [0 for i in range(super().__len__())]
        self.policy = self.inference_policy()

    def inference_policy(self):
        
        policy_list = []
        for idx in tqdm(
            range(
                super().__len__()
            )
        ):
            line = self.data[idx].strip().split()
            name = line[0]
            if self.train:
                img_path = self.file_path + "/train/" + name
            else:
                img_path = self.file_path + "/test/" + name
            img = Image.open(img_path).convert('RGB')
            if self.transform:
                transform = torchvision.transforms.Compose([
                    torchvision.transforms.Resize((112,112)),
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]
                    )
                ])
                img = transform(img)
            img = img.to(self.device)
            img = torch.reshape(img, (1, 3, 112, 112))
            self.policy_model.to(self.device)
            self.policy_model.eval()
            with torch.no_grad():
                policy = self.policy_model(img)

            # input_name = self.policy_model.get_inputs()[0].name
            # output_name = self.policy_model.get_outputs()[0].name
            # policy = self.policy_model.run(None, {input_name: img})


            policy = gumbel_softmax(policy, device=self.device)# [1, model_nums]
            policy = torch.argmax(policy, dim=-1)
            
            if self.train:            
                policy_list.append(name + " " + str(policy.item()) + "\n")
            else:            
                policy_list.append(name + " " + str(policy.item()) + "\n")
        return policy_list

    def cal_policy(self):
        if self.train:            
            with open(
                self.file_path + "/train_" + self.model_type + "_policy.txt", 
                'w'
            ) as f:
                f.write('')
        else:
            with open(
                self.file_path + "/test_" + self.model_type + "_policy.txt", 
                'w'
            ) as f:
                f.write('')
        
        model_list = []
        file_name_list = os.listdir("./proxynetworks/models/pth/" + self.model_type)
        file_name_list.sort()
        for model_name in file_name_list:
            model_list.append(
                torch.load(
                    "./proxynetworks/pth/" + self.model_type + "/" + model_name
                )
            )

        for idx in tqdm(
            range(
                super().__len__()
            )
        ):
            line = self.data[idx].strip().split()
            name = line[0]
            if self.train:
                img_path = self.file_path + "/train/" + name
            else:
                img_path = self.file_path + "/test/" + name
            img = Image.open(img_path).convert('RGB')
            if self.transform:
                img = self.transform(img)
            
            img = img.to(self.device)
            img = torch.reshape(img, (1, 3, 224, 224))
            # self.policy_model.to(self.device)
            # self.policy_model.eval()

            # policy = self.policy_model(img)

            input_name = self.policy_model.get_inputs()[0].name
            output_name = self.policy_model.get_outputs()[0].name
            policy = self.policy_model.run(None, {input_name: img})


            policy = gumbel_softmax(policy, device=self.device)# [1, model_nums]
            policy = torch.argmax(policy, dim=-1)



            if self.train:            
                with open(
                    self.file_path + "/train_" + self.model_type + "_policy.txt", 
                    'a'
                ) as f:
                    f.writelines(name + " " + str(policy.item()) + "\n")
            else:            
                with open(
                    self.file_path + "/test_" + self.model_type + "_policy.txt", 
                    'a'
                ) as f:
                    f.writelines(name + " " + str(policy.item()) + "\n")

    def read_policy(self):
        if self.train:
            with open(
                self.file_path + "/train_" + self.model_type + "_policy.txt", 
                'r'
            ) as f:
                self.policy = f.readlines()
        else:
            with open(self.file_path + "/test_" + self.model_type + "_policy.txt", 
                'r'
            ) as f:
                self.policy = f.readlines()

    def upd_diff(self):
        self.cal_policy(self)
        self.read_policy(self)

        print(self.policy)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        data, label = super().__getitem__(index)
        return index, data, label
    
    def getpolicy(self, index):
        line = self.policy[index].strip().split()
        policy = [int(x) for x in line[1:]]
        if self.policy_offset[index] >= 0:
            return (
                policy[0] + self.policy_offset[index]
            ) if (
                (policy[0] + self.policy_offset[index]) < self.model_nums) else (self.model_nums-1)
        else:
            return (
                policy[0] + self.policy_offset[index]
            ) if (
                (policy[0] + self.policy_offset[index]) >= 0) else 0
    
    def upd_policy_with_imp(
            self, 
            idx, 
            num
        ):
        self.policy_offset[idx] += num

    def upd_policy_label(self):
        for i in range(len(self.policy_label)):
            self.policy_label[i] = self.getpolicy(i)
        return self.policy_label
    
class Policy_Sampler(Sampler):
    def __init__(
            self, 
            data_source, 
            batch_size
        ):
        self.data_source = data_source
        self.batch_size = batch_size
        self.indices = list(range(len(self.data_source)))
        self.labels = [0 for i in range(len(self.indices))]
        self.indices.sort(key=lambda x: (self.labels[x], x))
        # self.shuffle()

    def upd_policy_with_imp(self, idx, num):
        self.data_source.upd_policy_with_imp(idx, num)

    def shuffle(self):
        indices = []
        for m in range(
            len(
                os.listdir(
                    "./proxynetworks/models/jit/" + self.data_source.model_type
                )
            )
        ):
            label_indices = [i for i in self.indices if self.labels[i] == m]
            random.shuffle(label_indices)
            indices += label_indices
        batch_indices = [
            int(i*self.batch_size) for i in range(
                int(len(self.data_source) / self.batch_size))
        ]
        random.shuffle(batch_indices)
        batch_indices_new = []
        for i in batch_indices:
            for j in range(self.batch_size):
                batch_indices_new.append(i+j)
        indices_new = [indices[i] for i in batch_indices_new]
        return indices_new + indices[len(indices):-1]

    def __iter__(self):
        if self.data_source.train:
            tmp = self.data_source.upd_policy_label()
            
        for idx in self.indices:
            self.labels[idx] = self.data_source.getpolicy(idx)
        self.indices.sort(key=lambda x: (self.labels[x], x))
        indices = self.shuffle()
        # if self.data_source.train:
        #     print([self.labels[idx] for idx in indices])
        #     print([tmp[idx] for idx in indices])
        return iter([i for i in indices])
    
    def __len__(self):
        return len(self.data_source)

class MNN_Dataset_imp(MNN_Dataset):
    def __init__(
            self, 
            file_path, 
            train, 
            transform, 
            num_classes
        ):
        super().__init__(
            file_path=file_path, 
            train=train, 
            transform=transform
        )
        self.num_classes = num_classes

    def __getitem__(self, index):
        data, label = super().__getitem__(index)
        return index, data, label

class Imp_Sampler(Sampler):
    def __init__(
            self, 
            data_source, 
            batch_size, 
            imp_threshold
        ):
        self.data_source = data_source
        self.batch_size = batch_size
        self.imp = [float('inf') for i in range(len(self.data_source))]
        self.indices = list(range(len(self.data_source)))
        self.imp_threshold = imp_threshold
        self.len = len(self.data_source)
    
    def upd_imp(
            self, 
            model, 
            device
        ):
        for index, data, label in tqdm(self.data_source):
            data = data.to(device)
            data = torch.reshape(data, (1, 3, 224, 224))
            label = torch.tensor([label])
            label = label.to(device)
            pred = model(data)
            loss = torch.nn.CrossEntropyLoss()(pred, label)
            self.imp[index] = loss.item()

    def __iter__(self):
        # random.shuffle(self.indices)
        # indices = [i for i in self.indices if self.imp[i] > self.imp_threshold]
        indices = self.indices
        # indices.sort(key=lambda x: (self.imp[x], x), reverse=True)
        # indices = indices[:int(3*len(indices)/4)]
        random.shuffle(indices)
        self.len = len(indices)
        return iter(indices)
    
    def __len__(self):
        return self.len