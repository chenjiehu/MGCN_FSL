import argparse
import os

import torch
import datetime
from torch.utils.data import DataLoader
from  dataset import DATASET

from samplers import CategoriesSampler
from convnet import Convnet
from Multi_gcn import MultiGCN
from utils import pprint, set_gpu, count_acc, Averager, euclidean_metric
from GCN_layers import GCN
from cluster_value import class_intra_inter
import torch.nn.functional as F
from utils import dataset_train, dataset_test, root, step1_save_path, step2_save_path

def train_with_k_shot(args, model, feature_shot, feature_query):
    #print('××××××××××××××××TSETING WITH test%s****************')

    model_GNN = model
    feature = torch.cat([feature_shot,feature_query])
    p = args.shot * args.way

    feature = model(feature)
    proto = feature[:p]
    proto = proto.reshape(args.shot, args.way, -1).mean(dim=0)

    label = torch.arange(args.way).repeat(round(feature_query.size(0)/args.way))
    label = label.type(torch.cuda.LongTensor)

    logits = euclidean_metric(feature[p:], proto)
    loss = F.cross_entropy(logits, label)
    return loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--load', default=step2_save_path)      ######  ProtoGCN  ProtoIGCN  MultiGCN ######      ########        ####finetune-epoch-8
    parser.add_argument('--batch', type=int, default=50)
    parser.add_argument('--way', type=int, default=10)
    parser.add_argument('--shot', type=int, default=10)
    parser.add_argument('--query', type=int, default=10)
    args = parser.parse_args()
    pprint(vars(args))

    set_gpu(args.gpu)

    dataset = DATASET(dataset_test)      ######  ######      ####### ######  ######      #######

    sampler = CategoriesSampler(dataset.label,
                                args.batch, args.way, args.query+args.shot)
    loader = DataLoader(dataset, batch_sampler=sampler,num_workers=4, pin_memory=True)

    model_CNN = Convnet().cuda()
    model_CNN.load_state_dict(torch.load(os.path.join(args.load,'CNNmax-acc.pth')))        ######  ######      ########  CNNfinetune-epoch-8
    #model_CNN.load_state_dict(torch.load('./save/proto-1/AID/10way_20shot/proto-max-acc.pth'))
    model = MultiGCN(input_dim=1600, N_way = args.way).cuda()           ######  ######      ######
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    model.load_state_dict(torch.load(os.path.join(args.load, 'max-acc.pth')))

    ave_acc = Averager()
    model_CNN.eval()
    model.eval()

    CNN_intra_inter_var = 0
    CNN_var_save = 0
    GCN_intra_inter_var = 0
    GCN_var_save = 0


    start_time = datetime.datetime.now()
    for i, batch in enumerate(loader,1):
        data, _ =[_.cuda() for _ in batch]
        kk = args.shot * args.way
        # model.load_state_dict(torch.load(args.load))
        feature = model_CNN(data)              #计算proto中心

        proto = feature[:kk]
        proto = proto.reshape(args.shot, args.way, -1).mean(dim=0)

        model.eval()

        #############################################################################################

        label = torch.arange(args.way).repeat(args.query)   #扩展标签
        label = label.type(torch.cuda.LongTensor)

        ###########################################################################################################
        feature = model(feature)
        proto = feature[:kk]
        proto = proto.reshape(args.shot, args.way, -1).mean(dim=0)
        logits = euclidean_metric(feature[kk:], proto)

        label = torch.arange(args.way).repeat(args.query)
        label = label.type(torch.cuda.LongTensor)

        acc = count_acc(logits, label)
        ave_acc.add(acc)
        print('batch {}: {:.2f}({:.2f})'.format(i, ave_acc.item() * 100, acc * 100))
        x = None;
        p = None;
        logits = None
    end_time = datetime.datetime.now()
    print(end_time - start_time)

if __name__ == '__main__':
    main()
