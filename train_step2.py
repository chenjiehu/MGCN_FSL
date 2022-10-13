import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import argparse
import os.path as osp
import time
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataset import DATASET

from samplers import CategoriesSampler
from convnet import Convnet
from Multi_gcn import  MultiGCN
from utils import pprint, set_gpu, ensure_path, Averager, Timer, count_acc, euclidean_metric, cos_metric
from utils import dataset_train, dataset_val, root, step1_save_path, step2_save_path

if __name__ == '__main__':
    parser = argparse.ArgumentParser()  # 初始化参数
    parser.add_argument('--max-epoch', type=int, default=30)
    parser.add_argument('--save-epoch', type=int, default=20)
    parser.add_argument('--shot', type=int, default=5)
    parser.add_argument('--query', type=int, default=15)
    parser.add_argument('--train-way', type=int, default=10)
    parser.add_argument('--test-way', type=int, default=10)
    parser.add_argument('--load-path', default=step1_save_path)
    parser.add_argument('--save-path', default=step2_save_path)      ######  ######      ########        ####
    parser.add_argument('--gpu', default='0')
    args = parser.parse_args()
    pprint(vars(args))

    set_gpu(args.gpu)
    ensure_path(args.save_path)

    trainset = DATASET(dataset_train)         ######  ######      ########        ####
    train_sampler = CategoriesSampler(trainset.label, 100,
                                      args.train_way, args.shot + args.query)
    train_loader = DataLoader(dataset=trainset, batch_sampler=train_sampler,
                              num_workers=8, pin_memory=True)

    valset = DATASET(dataset_val)         ######  ######      ########        ####
    val_sampler = CategoriesSampler(valset.label, 100,
                                    args.test_way, args.shot + args.query)
    val_loader = DataLoader(dataset=valset, batch_sampler=val_sampler,
                            num_workers=8, pin_memory=True)
    model_CNN = Convnet().cuda()
    model_CNN.load_state_dict(torch.load(os.path.join(args.load_path,'proto-max-acc.pth'))) ###此处需要与train.py存储参数的路径相同
    optimizer_CNN = torch.optim.Adam(model_CNN.parameters(),lr=0.001)

    model = MultiGCN(input_dim=1600, N_way=10).cuda()     ###########
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)


    def save_model(name):
        torch.save(model_CNN.state_dict(), osp.join(args.save_path, 'CNN' + name + '.pth'))
        torch.save(model.state_dict(), osp.join(args.save_path, name + '.pth'))


    trlog = {}
    trlog['args'] = vars(args)
    trlog['train_loss'] = []
    trlog['val_loss'] = []
    trlog['train_acc'] = []
    trlog['val_acc'] = []
    trlog['max_acc'] = 0.0

    timer = Timer()

    for epoch in range(1, args.max_epoch + 1):
        lr_scheduler.step()  # 补偿自动调整
        model_CNN.train()
        model.train()  # 模型开始训练

        tl = Averager()
        ta = Averager()

        start_time = time.time()
        for i, batch in enumerate(train_loader, 1):
            data, _ = [_.cuda() for _ in batch]
            p = args.shot * args.train_way
            data_shot, data_query = data[:p], data[p:]  # 选取每个batch的前p列作为suppot集，后面作为查询级

            feature = model_CNN(data)
            feature = model(feature)
            proto = feature[:p]# CNN提取特征
            proto = proto.reshape(args.shot, args.train_way, -1)
            proto = proto.mean(dim = 0)

            label = torch.arange(args.train_way).repeat(args.query)
            label = label.type(torch.cuda.LongTensor)

            logits = euclidean_metric(feature[p:], proto)  # 利用欧式距离来判断标签个数
            loss = F.cross_entropy(logits, label)
            acc = count_acc(logits, label)
            print('epoch {}, train {}/{}, loss={:.4f} acc={:.4f}'
                   .format(epoch, i, len(train_loader), loss.item(), acc))

            tl.add(loss.item())
            ta.add(acc)

            optimizer_CNN.zero_grad()
            optimizer.zero_grad()
            loss.backward()
            optimizer_CNN.step()
            optimizer.step()

            proto = None;
            logits = None;
            loss = None

        end_time = time.time()
        print(end_time-start_time)
        tl = tl.item()
        ta = ta.item()

        model_CNN.eval()
        model.eval()

        vl = Averager()
        va = Averager()

        for i, batch in enumerate(val_loader, 1):
            data, _ = [_.cuda() for _ in batch]
            p = args.shot * args.test_way
            data_shot, data_query = data[:p], data[p:]

            feature = model_CNN(data)
            feature = model(feature)
            proto = feature[:p]
            proto = proto.reshape(args.shot, args.test_way, -1).mean(dim=0)

            label = torch.arange(args.test_way).repeat(args.query)
            label = label.type(torch.cuda.LongTensor)

            logits = euclidean_metric(feature[p:], proto)
            loss = F.cross_entropy(logits, label)
            acc = count_acc(logits, label)

            vl.add(loss.item())
            va.add(acc)

            proto = None;
            logits = None;
            loss = None

        vl = vl.item()
        va = va.item()
        print('epoch {}, val, loss={:.4f} acc={:.4f}'.format(epoch, vl, va))

        if va > trlog['max_acc']:
            trlog['max_acc'] = va
            save_model('max-acc')

        trlog['train_loss'].append(tl)
        trlog['train_acc'].append(ta)
        trlog['val_loss'].append(vl)
        trlog['val_acc'].append(va)

        torch.save(trlog, osp.join(args.save_path, 'trlog_10way_5shot'))####################################

        save_model('finetune-epoch-last')

        if epoch % args.save_epoch == 0:
            save_model('finetune-epoch-{}'.format(epoch))

        print('ETA:{}/{}'.format(timer.measure(), timer.measure(epoch / args.max_epoch)))

