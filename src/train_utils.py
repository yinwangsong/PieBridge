import random
import torch
from tqdm import tqdm
import sys
from MNN_dataset import MNN_Dataset, MNN_Dataset_policy, MNN_Dataset_diff, Policy_Sampler, Diff_Sampler, Imp_Sampler, MNN_Dataset_imp
import time
from edgemodel_utils import ModelWrapper3

def select_bkb(
        idx, 
        select, 
        model_nums, 
        epoch, 
        device
    ):
    def are_lists_equal(list1, a):
        for i in range(len(list1)):
            if list1[i] != a:
                return False
        return True
    
    if are_lists_equal(idx, idx[0]):
        return idx[0]
    else:
        return idx[0]
        # exit()

@torch.no_grad()
def evaluate_head(
        backbones, 
        head, 
        data_loader, 
        model_nums, 
        select, 
        epoch, 
        device
    ):
    sum_num = torch.zeros(1, dtype=torch.int).to(device)
    data_loader = tqdm(data_loader, file=sys.stdout)

    for step, data in enumerate(data_loader):
        index, images, labels = data
        backbone = backbones[0]
        backbone.to(device)
        head.to(device)
        backbone.eval()
        head.eval()

        # input_name = backbone.get_inputs()[0].name
        # output_name = backbone.get_outputs()[0].name

        # tmp = backbone.run(None, {input_name: images.cpu().numpy()})[0]
        tmp = backbone(images.to(device))
        pred = head(
            tmp[0]
        )
        pred = torch.max(pred, dim=1)[1]
        sum_num += torch.eq(
            pred, 
            labels.to(device)
        ).sum()
    return sum_num

@torch.no_grad()
def evaluate_cos(
        backbone1, 
        backbone2, 
        data_loader, 
        device
    ):
    backbone1.to(device)
    backbone2.to(device)
    backbone1.eval()
    backbone2.eval()

    data_loader = tqdm(data_loader, file=sys.stdout)

    similarity_all = torch.zeros(1, dtype=torch.float).to(device)
    
    cnt = 0
    simi_list = []
    for step, data in enumerate(data_loader):
        cnt += 1
        images, labels, policy = data
        pred1 = backbone1(images.to(device))
        pred2 = backbone2(images.to(device))

        cos = torch.nn.CosineSimilarity(dim=1)
        similarity = cos(pred1, pred2)
        similarity_batch = torch.mean(similarity, dim=0)
        similarity_all += similarity_batch
        simi_list += similarity.tolist()

    return similarity_all / cnt

def train_one_epoch_multibkb_head(
        backbones, 
        head, 
        train_data_loader, 
        test_data_loader, 
        train_dataset, 
        test_dataset, 
        device, 
        epoch, 
        sampler, 
        EMA, 
        pu, 
        log_pth,
        model_nums,
        lr, 
        time0, 
        sum_eval_time, 
        EMAalpha, 
        select=0
    ):

    loss_function = torch.nn.CrossEntropyLoss()
    bs = train_data_loader.batch_size
    train_data_loader = tqdm(train_data_loader, file=sys.stdout)

    bkb_idx_list = ""
    bp_list = ""

    for step, data in enumerate(train_data_loader):
        indexes, images, labels = data
        policy = []
        for i in range(len(indexes)):
            policy.append(train_dataset.policy_label[indexes[i]])
        idx = select_bkb(
            idx=policy, 
            device=device, 
            select=select, 
            model_nums=model_nums, 
            epoch=epoch
        )
        # optimizer = torch.optim.SGD(head.parameters(), lr=lr)
        optimizer = torch.optim.SGD(head.parameters(), lr=lr)#/(2**idx))
        backbone = backbones[idx]

        bkb_idx_list += " " + str(idx)

        backbone.eval()
        head.eval()
        backbone.to(device)
        # tmp = backbone(images.to(device))

        # from time import time

        # time0 = time()
        # input_name = backbone.get_inputs()[0].name
        # output_name = backbone.get_outputs()[0].name
        # print(type(images.to(device)))
        # tmp = backbone.run(None, {input_name: images.cpu().numpy()})
        with torch.no_grad():
            tmp = backbone(images.to(device))
        # print(time()-time0)
        # x = tmp[0].detach()
        # time0 = time()
        # x = torch.from_numpy(tmp[0]).detach().to(device)
        x = tmp[0]
        # print(time()-time0)
        # time0 = time()
        pred = head(x)
        # print(time()-time0)
        loss = loss_function(pred, labels.to(device))
        train_data_loader.desc = "[epoch {}] mean loss {}".format(epoch, round(loss.item(), 3))
        if loss.item() > EMAalpha*EMA:
            bp_list += " " + str(1)
            loss.backward()
            optimizer.step()
            for i in range(indexes.size(0)):
                sampler.upd_policy_with_imp(indexes[i].item(), 0)
        else:
            bp_list += " " + str(0)
            for i in range(indexes.size(0)):
                sampler.upd_policy_with_imp(indexes[i].item(), 1)

        if EMA == -1:
            EMA = loss.item()
        else:
            EMA = pu * loss.item() + (1-pu) * EMA

        optimizer.zero_grad()

        if ((step+1) % (int(len(train_dataset)/bs)-1)) == 0:
            tmptime0 = time.time()
            sum_num = evaluate_head(
                backbones=backbones, 
                head=head, 
                data_loader=test_data_loader,
                model_nums=model_nums,
                select=1,
                epoch=0,
                device=device
            )
            tmptime1 = time.time()
            sum_eval_time += (tmptime1 - tmptime0)
            acc = sum_num / torch.FloatTensor([len(test_dataset)]).to(device)
            # with open(log_pth, 'a') as f:
            #     f.write(str(acc.item()) + bkb_idx_list + "\n")
            #     bkb_idx_list = ""
            # with open(log_pth + '.txt', 'a') as f:
            #     f.write(str(acc.item()) + bp_list + "\n")
            #     bp_list = ""
            with open(log_pth, 'a') as f:
                f.write(str(acc.item()) + " " + str(time.time() - time0 - sum_eval_time) + "\n")
    return bkb_idx_list, EMA, sampler, sum_eval_time

def train_multibkb_head(
        backbones, 
        head, 
        train_dataset, 
        test_dataset, 
        epochs, 
        device, 
        log_pth, 
        model_nums, 
        lr, 
        bs, 
        EMAalpha, 
        select=0
    ):
    # for bkb in backbones:
    #     bkb.to(device)
    head.to(device)

    train_sampler = Policy_Sampler(
        data_source=train_dataset, 
        batch_size=bs
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=bs, 
        shuffle=False, 
        num_workers=1, 
        pin_memory=True, 
        drop_last=True, 
        sampler=train_sampler
    )
    test_sampler = Policy_Sampler(
        data_source=test_dataset, 
        batch_size=bs
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, 
        batch_size=4, 
        shuffle=False, 
        num_workers=1, 
        pin_memory=True, 
        drop_last=True, 
        sampler=test_sampler
    )

    # with open(log_pth, 'w') as f:
    #     f.write("")
    # with open(log_pth+'.txt', 'w') as f:
    #     f.write("")
    with open(log_pth, 'w') as f:
        f.write("")

    EMA = -1
    pu = 2/(10+1)

    time0 = time.time()

    sum_eval_time = 0.0
    
    for epoch in range(epochs):
        bkb_idx_list, EMA, train_sampler, sum_eval_time = train_one_epoch_multibkb_head(
            backbones=backbones, 
            head=head, 
            train_data_loader=train_loader, 
            test_data_loader=test_loader, 
            test_dataset=test_dataset, 
            device=device, 
            epoch=epoch, 
            log_pth=log_pth, 
            select=select, 
            model_nums=model_nums,
            lr=lr,
            EMA=EMA,
            pu=pu,
            sampler=train_sampler,
            train_dataset=train_dataset,
            time0 = time0,
            sum_eval_time = sum_eval_time,
            EMAalpha = EMAalpha
        )

        # sum_num = evaluate_head(
        #     backbones=backbones, 
        #     head=head, 
        #     data_loader=test_loader,
        #     model_nums=model_nums,
        #     select=select,
        #     epoch=0, 
        #     device=device
        # )
        # acc = sum_num / torch.FloatTensor([len(test_dataset)]).to(device)
        # with open(log_pth, 'a') as f:
        #     f.write(str(acc.item())+bkb_idx_list+"\n")
        # print("[epoch {}] accuracy: {}".format(epoch, round(acc.item(), 5)))

        train_loader = torch.utils.data.DataLoader(
            train_dataset, 
            batch_size=bs, 
            shuffle=False, 
            num_workers=1, 
            pin_memory=True, 
            drop_last=True, 
            sampler=train_sampler
        )

@torch.no_grad()
def evaluate_adapter(
        backbones, 
        data_loader, 
        model_nums, 
        select, 
        epoch, 
        device
    ):
    sum_num = torch.zeros(1, dtype=torch.int).to(device)

    data_loader = tqdm(data_loader, file=sys.stdout)

    for step, data in enumerate(data_loader):
        indexes, images, labels = data
        
        idx = select_bkb(
            idx=[0], 
            device=device, 
            select=select, 
            model_nums=model_nums, 
            epoch=epoch
        )
        backbone = backbones[idx]
        backbone.to(device)
        backbone.eval()

        pred = backbone(images.to(device))
        pred = torch.max(pred, dim=1)[1]
        sum_num += torch.eq(pred, labels.to(device)).sum()
    return sum_num

@torch.no_grad()
def evaluate_adapter_imp(
        backbones, 
        data_loader, 
        model_nums, 
        select, 
        epoch, 
        device
    ):
    sum_num = torch.zeros(1, dtype=torch.int).to(device)

    data_loader = tqdm(data_loader, file=sys.stdout)

    for step, data in enumerate(data_loader):
        indexes, images, labels = data
        
        idx = select_bkb(
            idx=0, 
            device=device, 
            select=select, 
            model_nums=model_nums, 
            epoch=epoch
        )
        backbone = backbones[idx]
        backbone.to(device)
        backbone.eval()

        pred = backbone(images.to(device))
        pred = torch.max(pred, dim=1)[1]
        sum_num += torch.eq(pred, labels.to(device)).sum()
    return sum_num

def train_one_epoch_multibkb_adapter(
        backbones, 
        train_data_loader, 
        test_data_loader, 
        train_dataset, 
        test_dataset, 
        device, 
        epoch, 
        log_pth,
        model_nums, 
        lr, 
        EMA, 
        pu, 
        sampler, 
        EMAalpha, 
        time0, 
        sum_eval_time, 
        select=0
    ):
    loss_function = torch.nn.CrossEntropyLoss()
    bs = train_data_loader.batch_size

    train_data_loader = tqdm(train_data_loader, file=sys.stdout)

    bkb_idx_list = ""
    bp_list = ""

    adapters, head = backbones[0].export_public_parts()

    tmp = []
    tmp2 = []
    nums1 = 0
    nums2 = 0
    for i in train_dataset.policy_label:
        if i != 0:
            nums1 += 1
    for step, data in enumerate(train_data_loader):
        indexes, images, labels = data
        policy = []
        for i in range(len(indexes)):
            policy.append(train_dataset.policy_label[indexes[i]])
        tmp += policy
        tmp2 += list(indexes.numpy())
        idx = select_bkb(
            idx=policy, 
            device=device, 
            select=select, 
            model_nums=model_nums, 
            epoch=epoch
        )
        backbone = backbones[idx]

        # optimizer = torch.optim.SGD(
        #     filter(
        #         lambda p: p.requires_grad, 
        #         backbone.parameters()
        #     ), 
        #     lr=lr
        # )
        optimizer = torch.optim.Adam(
            filter(
                lambda p: p.requires_grad, 
                backbone.parameters()
            ), 
            lr=lr/(10**(idx))
        )

        bkb_idx_list += " " + str(idx)
        backbone.eval()

        backbone.upd_public_parts(adapters, head)

        pred = backbone(images.to(device))
        loss = loss_function(pred, labels.to(device))
        train_data_loader.desc = "[epoch {}] mean loss {}".format(epoch, round(loss.item(), 3))
        if loss.item() > EMAalpha*EMA:
            bp_list += " " + str(1)
            loss.backward()
            optimizer.step()
            for i in range(indexes.size(0)):
                sampler.upd_policy_with_imp(indexes[i].item(), 0)
        else:
            bp_list += " " + str(0)
            for i in range(indexes.size(0)):
                sampler.upd_policy_with_imp(indexes[i].item(), 1)

        if EMA == -1:
            EMA = loss.item()
        else:
            EMA = pu * loss.item() + (1-pu) * EMA
        optimizer.zero_grad()

        adapters, head = backbone.export_public_parts()

        if ((step+1) % (int(len(train_dataset)/bs)-1)) == 0:
            tmptime0 = time.time()
            sum_num = evaluate_adapter(
                backbones=backbones, 
                data_loader=test_data_loader,
                model_nums=model_nums,
                select=1,
                epoch=0,
                device=device
            )
            tmptime1 = time.time()
            sum_eval_time += (tmptime1 - tmptime0)
            acc = sum_num / torch.FloatTensor([len(test_dataset)]).to(device)
            # with open(log_pth, 'a') as f:
            #     f.write(str(acc.item())+bkb_idx_list+"\n")
            #     bkb_idx_list = ""
            # with open(log_pth+'.txt', 'a') as f:
            #     f.write(str(acc.item())+bp_list+"\n")
            #     bp_list = ""
            with open(log_pth, 'a') as f:
                f.write(str(acc.item())+" "+str(time.time()-time0-sum_eval_time)+"\n")
    return bkb_idx_list, EMA, sampler, sum_eval_time

def train_multibkb_adapter(
        backbones, 
        train_dataset, 
        test_dataset, 
        epochs, 
        device, 
        log_pth, 
        model_nums, 
        lr, 
        bs, 
        EMAalpha, 
        select=0
    ):
    for bkb in backbones:
        bkb.to(device)

    train_sampler = Policy_Sampler(
        data_source=train_dataset, 
        batch_size=bs
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=bs, 
        shuffle=False, 
        num_workers=1, 
        pin_memory=True, 
        drop_last=True, 
        sampler=train_sampler
    )
    test_sampler = Policy_Sampler(
        data_source=test_dataset, 
        batch_size=bs
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, 
        batch_size=4, 
        shuffle=False, 
        num_workers=1, 
        pin_memory=True, 
        drop_last=True,
        sampler=test_sampler
    )

    # with open(log_pth, 'w') as f:
    #     f.write("")
    # with open(log_pth+'.txt', 'w') as f:
    #     f.write("")
    with open(log_pth, 'w') as f:
        f.write("")
    # 进行训练

    EMA = -1
    pu = 2/(10+1)

    time0 = time.time()

    sum_eval_time = 0.0

    for epoch in range(epochs):
        
        bkb_idx_list, EMA, train_sampler, sum_eval_time = train_one_epoch_multibkb_adapter(
            backbones=backbones, 
            train_data_loader=train_loader, 
            test_data_loader=test_loader, 
            test_dataset=test_dataset, 
            device=device, 
            epoch=epoch, 
            log_pth=log_pth, 
            select=select, 
            model_nums=model_nums,
            lr=lr,
            EMA=EMA,
            pu=pu,
            sampler=train_sampler,
            train_dataset=train_dataset,
            EMAalpha=EMAalpha,
            time0=time0,
            sum_eval_time=sum_eval_time
        )

        sum_num = evaluate_adapter(
            backbones=backbones, 
            data_loader=test_loader,
            model_nums=model_nums,
            select=select,
            epoch=0, 
            device=device
        )
        acc = sum_num / torch.FloatTensor([len(test_dataset)]).to(device)
        # with open(log_pth, 'a') as f:
        #     f.write(str(acc.item())+bkb_idx_list+"\n")

        print("[epoch {}] accuracy: {}".format(epoch, round(acc.item(), 5)))

        train_loader = torch.utils.data.DataLoader(
            train_dataset, 
            batch_size=bs, 
            shuffle=False, 
            num_workers=1, 
            pin_memory=True, 
            drop_last=True, 
            sampler=train_sampler
        )

def train_onebkb_all(
        train_dataset, 
        test_dataset, 
        epochs, 
        device, 
        log_pth, 
        lr, 
        bs, 
        model, 
        is_prune, 
        is_elastic, 
        is_lastk, 
        model_name
    ):
    kernel_weights = []
    for name, w in model.named_parameters():
        if 'weight' in name:
            kernel_weights.append(w)
            w.requires_grad = True
        else:
            if is_prune:
                w.requires_grad = False
            else:
                w.requires_grad = True
    
    if model_name == 'vit':
        if is_elastic:
            pos = int(0.5*len(model.backbone.blocks))
            model = ModelWrapper3(model=model, pos=pos, model_name=model_name)
        if is_prune:
            pos = int(0.0*len(model.backbone.blocks))
            model = ModelWrapper3(model=model, pos=pos, model_name=model_name)
    else:
        if is_elastic:
            pos = int(0.5*len(model.backbone.model.blocks))
            model = ModelWrapper3(model=model, pos=pos, model_name=model_name)
        if is_prune:
            pos = int(0.0*len(model.backbone.model.blocks))
            model = ModelWrapper3(model=model, pos=pos, model_name=model_name)
        if is_lastk:
            pos = int(0.75*len(model.backbone.model.blocks))
            model = ModelWrapper3(model=model, pos=pos, model_name=model_name)

    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         print(name)
    model.to(device)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=bs, 
        shuffle=True, 
        num_workers=1, 
        pin_memory=True, 
        drop_last=True,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, 
        batch_size=4, 
        shuffle=False,
        num_workers=1,
        pin_memory=True,
        drop_last=True
    )
    loss_function = torch.nn.CrossEntropyLoss()
    if model_name=='vit':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=lr
        )
    else:
        optimizer = torch.optim.SGD(
            model.parameters(), 
            lr=lr
        )
    with open(log_pth, 'w') as f:
        f.write("")
    time0 = time.time()
    sum_eval_time = 0.0
    for epoch in range(epochs):
        train_data_loader = tqdm(train_loader, file=sys.stdout)
        model.eval()
        for step, data in enumerate(train_data_loader):
            images, labels = data
            # from time import time
            # time0 = time()
            pred = model(images.cuda())
            # print(time()-time0)
            # print(pred.shape)
            if is_prune:
                L1_penalty = 1e-4 * torch.sum(
                    torch.Tensor(
                        [
                            torch.sum(torch.abs(w)) for w in kernel_weights
                        ]
                    ).to(device)
                )
                loss = loss_function(pred, labels.to(device)) + L1_penalty
            else:
                loss = loss_function(pred, labels.to(device))
            train_data_loader.desc = "[epoch {}] mean loss {}".format(epoch, round(loss.item(), 3))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            if (((step+1) % (int(len(train_dataset)/bs) - 1))) == 0:
                tmptime0 = time.time()
                sum_num = torch.zeros(1, dtype=torch.int).to(device)
                test_data_loader = tqdm(test_loader, file=sys.stdout)
                for step, data in enumerate(test_data_loader):
                    images, labels = data
                    pred = model(images.cuda())
                    pred = torch.max(pred, dim=1)[1]
                    sum_num += torch.eq(pred, labels.to(device)).sum()
                    acc = sum_num / torch.FloatTensor([len(test_dataset)]).to(device)
                tmptime1 = time.time()
                sum_eval_time += (tmptime1 - tmptime0)
                with open(log_pth, 'a') as f:
                    f.write(str(acc.item()) + " " + str(time.time() - time0-sum_eval_time) + "\n")