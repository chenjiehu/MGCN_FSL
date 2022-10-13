A re-implementation of Multi-GCN


## Environment

* python 3.6
* pytorch 0.4.0
* pycharm

## Instructions

Before running you program, the path of the dataset may need to be modified on your computer

1 For FSL of the same dataset, please follow these steps to perform the program:

(1) run train_step1.py (pretrain Conv4)
(2) run train_step2.py (train Multi-GCN)
(3) run test_multiGCN.py 

2 For FSL of the cross datasets, please follow these steps to perform the program:

(1) run train_2class.py (pretrain Conv4)
(2) run train_multiGCN_2class.py (train Multi-GCN)
(3) run test_multiGCN_2class.py 

You can test the classification accuracy in different situations by modifying the relevant parameters
train-way,test-way,shot



