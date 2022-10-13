import os
import shutil
import time
import pprint

import torch
root = './'
dataset_train = 'data/train'   # auxiliary data
dataset_val = 'data/val'    #val data
dataset_test  ='data/test'   #test data
step1_save = 'save/step1'  #CNN pre-train
step1_save_path = os.path.join(root, step1_save)
step2_save = 'save/step2'  # multiGNN train
step2_save_path = os.path.join(root, step2_save)


def set_gpu(x):
    os.environ['CUDA_VISIBLE_DEVICES'] = x
    print('using gpu:', x)


def ensure_path(path):
    if os.path.exists(path):
        if input('{} exists, remove? ([y]/n)'.format(path)) != 'n':
            shutil.rmtree(path)
            os.makedirs(path)
    else:
        os.makedirs(path)


class Averager():

    def __init__(self):
        self.n = 0
        self.v = 0

    def add(self, x):
        self.v = (self.v * self.n + x) / (self.n + 1)
        self.n += 1

    def item(self):
        return self.v


def count_acc(logits, label):
    pred = torch.argmax(logits, dim=1)
    return (pred == label).type(torch.cuda.FloatTensor).mean().item()


def dot_metric(a, b):
    return torch.mm(a, b.t())


def euclidean_metric(a, b):
    n = a.shape[0]
    m = b.shape[0]
    a = a.unsqueeze(1).expand(n, m, -1)
    b = b.unsqueeze(0).expand(n, m, -1)
    logits = -((a - b)**2).sum(dim=2)
    return logits


def cos_metric(a, b):
    n = a.shape[0]
    m = b.shape[0]
    a = a.unsqueeze(1).expand(n, m, -1)
    b = b.unsqueeze(0).expand(n, m, -1)
    ab = torch.mul(a,b)
    ab = torch.sum(ab, dim=2)
    a_norm = torch.norm(a,dim=2)
    b_norm = torch.norm(b,dim=2)
    ab_norm = torch.mul(a_norm,b_norm)
    logits = ab/ab_norm
    return logits


def K_euclidean_metric(a, b, k, shot):
    n = a.shape[0]
    m = b.shape[0]
    a = a.unsqueeze(1).expand(n, m, -1)
    b = b.unsqueeze(0).expand(n, m, -1)
    logits = -((a - b)**2).sum(dim=2)
    #logits_e = torch.exp(logits/100)
    logits_e = logits
    logits_zeros = torch.zeros_like(logits_e)
    _, index = torch.topk(logits, k=k, dim=1, largest=True, sorted=False)
    for num in range(logits_zeros.size(0)):
        logits_zeros[num, index[num,:]] = 1
    logits = torch.mul(logits_e, logits_zeros)

    logits2 = logits.reshape(n,shot,-1).sum(dim=1)
    return logits2


class Timer():

    def __init__(self):
        self.o = time.time()

    def measure(self, p=1):
        x = (time.time() - self.o) / p
        x = int(x)
        if x >= 3600:
            return '{:.1f}h'.format(x / 3600)
        if x >= 60:
            return '{}m'.format(round(x / 60))
        return '{}s'.format(x)

_utils_pp = pprint.PrettyPrinter()
def pprint(x):
    _utils_pp.pprint(x)


def l2_loss(pred, label):
    return ((pred - label)**2).sum() / len(pred) / 2

