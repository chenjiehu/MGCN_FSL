#########################################################################
#########      该文件用来衡量特征向量的聚散程度              #################
########################################################################

import torch

def func(n):
    if n == 0 or n == 1:
        return 1
    else:
        return (n * func(n - 1))

def class_intra_value(m,h):#本类节点特征到本类原型的平均距离
#m是聚类中心特征向量拼接成的矩阵，ｈ是每个节点的特征向量拼接成的矩阵
    N_way=m.size(0)
    query_num=h.size(0)
    query = query_num//N_way
    h_reshape = h.reshape(query,N_way,-1)

    intra_class_var = torch.zeros(N_way).cuda()

    for i in range(N_way):
        for j in range(query):
            intra_class_var[i]=intra_class_var[i] + torch.norm(m[i,:]-h_reshape[j,i,:])
        intra_class_var[i] = intra_class_var[i]/query

    var_sum = torch.sum(intra_class_var)
    return var_sum/N_way

def class_inter_value(m):#类别之间原型的平均距离

    m_num = m.size(0)
    number = 0

    class_inter_var = torch.zeros(1).cuda()
    for i in range(m_num):
        for j in range(i+1,m_num):
            class_inter_var = class_inter_var + torch.norm(m[i,:]-m[j,:])
            number = number + 1

    return class_inter_var/number

def node_to_other_class(m,h):#本类节点特征到本类原型的平均距离
#m是聚类中心特征向量拼接成的矩阵，ｈ是每个节点的特征向量拼接成的矩阵
    N_way=m.size(0)
    query_num=h.size(0)
    query = query_num//N_way
    h_reshape = h.reshape(query,N_way,-1)

    intra_class_var = torch.zeros(N_way).cuda()

    for i in range(N_way):
        for j in range(query):
            for k in range(N_way):
                if k != i:
                    intra_class_var[i]=intra_class_var[i] + torch.norm(m[k,:]-h_reshape[j,i,:])
        intra_class_var[i] = intra_class_var[i]/(query*(N_way-1))

    var_sum = torch.sum(intra_class_var)
    return var_sum/N_way

def class_intra_inter(m,h):
    intra_class_var = class_intra_value(m,h)

    inter_class_var = node_to_other_class(m,h)

    return inter_class_var/intra_class_var

if __name__=='__main__':
    m = torch.randn(2,3).cuda()
    n = torch.randn(4,3).cuda()

    print(m)
    print(n)

    intra_var = class_intra_value(m,n)
    print(intra_var)
    inter_var = node_to_other_class(m,n)
    print(inter_var)


