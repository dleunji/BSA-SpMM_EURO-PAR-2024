#!/usr/bin/env python3
import torch
import numpy as np
import time

from config import *
from scipy.sparse import *


class TCGNN_dataset(torch.nn.Module):
    """
    data loading for more graphs
    """
    def __init__(self, path, dim, num_class, load_from_txt=True, verbose=False):
        super(TCGNN_dataset, self).__init__()

        # self.nodes = set()

        # self.load_from_txt = load_from_txt
        self.num_nodes = 0
        self.num_features = dim 
        self.num_classes = num_class
        self.edge_index = None
        
        self.reorder_flag = False
        self.verbose_flag = verbose

        self.avg_degree = -1
        self.avg_edgeSpan = -1

        self.init_edges(path)
        self.init_embedding(dim)
        self.init_labels(num_class)

        # train = 1
        # val = 0.3
        # test = 0.1
        # self.train_mask = [1] * int(self.num_nodes * train) + [0] * (self.num_nodes  - int(self.num_nodes * train))
        # self.val_mask = [1] * int(self.num_nodes * val)+ [0] * (self.num_nodes  - int(self.num_nodes * val))
        # self.test_mask = [1] * int(self.num_nodes * test) + [0] * (self.num_nodes  - int(self.num_nodes * test))
        # self.train_mask = torch.BoolTensor(self.train_mask).cuda()
        # self.val_mask = torch.BoolTensor(self.val_mask).cuda()
        # self.test_mask = torch.BoolTensor(self.test_mask).cuda()

    def init_edges(self, path):
        fp = open(path, "r")
        line = fp.readline().rstrip("\n")
        meta_data = [int(x) for x in line.split(",")]
        self.rows = meta_data[0]
        self.cols = meta_data[1]
        self.num_nodes = self.rows
        self.num_edges = meta_data[2]

        line = fp.readline().rstrip("\n")
        self.row_pointers = [int(x) for x in line.split(" ") if x != ""]
        
        line = fp.readline().rstrip("\n")
        self.column_index = [int(x) for x in line.split(" ") if x != "" ]
        
        self.row_pointers = torch.IntTensor(self.row_pointers)
        self.column_index = torch.IntTensor(self.column_index)

        # Get degrees array.
        degrees = (self.row_pointers[1:] - self.row_pointers[:-1]).tolist()
        self.degrees = torch.sqrt(torch.FloatTensor(list(map(func, degrees)))).cuda()

        if self.verbose_flag:
            print("# rows: {}".format(self.rows))
            print("# cols: {}".format(self.cols))

            # print('# nodes: {}'.format(self.num_nodes))
            # print("# avg_degree: {:.2f}".format(self.avg_degree))
            # print("# avg_edgeSpan: {}".format(int(self.avg_edgeSpan)))

    def init_embedding(self, dim):
        '''
        Generate node embedding for nodes.
        Called from __init__.
        '''
        self.x = torch.randn(self.num_nodes, dim).cuda()
    
    def init_labels(self, num_class):
        '''
        Generate the node label.
        Called from __init__.
        '''
        self.y = torch.ones(self.num_nodes).long().cuda()
