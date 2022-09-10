'''
MIT License

Copyright (c) 2021 Stephen Hausler, Sourav Garg, Ming Xu, Michael Milford and Tobias Fischer

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

We thank Nanne https://github.com/Nanne/pytorch-NetVlad for the original design of the NetVLAD
class which in itself was based on https://github.com/lyakaap/NetVLAD-pytorch/blob/master/netvlad.py
In our version we have significantly modified the code to suit our Patch-NetVLAD approach.
'''


# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from sklearn.neighbors import NearestNeighbors
# import faiss
# import numpy as np

from sklearn import neighbors
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors
import numpy as np

# graphsage
import torch.nn.init as init


class NeighborAggregator(nn.Module):
    def __init__(self, input_dim, output_dim,
                 use_bias=False, aggr_method="mean"):
        """Aggregate node neighbors

        Args:
            input_dim: the dimension of the input feature
            output_dim: the dimension of the output feature
            use_bias: whether to use bias (default: {False})
            aggr_method: neighbor aggregation method (default: {mean})
        """
        super(NeighborAggregator, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_bias = use_bias
        self.aggr_method = aggr_method
        self.weight = nn.Parameter(torch.Tensor(input_dim, output_dim))
        if self.use_bias:
            self.bias = nn.Parameter(torch.Tensor(self.output_dim))
        self.reset_parameters()
    
    def reset_parameters(self):
        init.kaiming_uniform_(self.weight)
        if self.use_bias:
            init.zeros_(self.bias)

    def forward(self, neighbor_feature):
       # print(neighbor_feature.shape)
        if self.aggr_method == "mean":
            aggr_neighbor = neighbor_feature.mean(dim=1)
        elif self.aggr_method == "sum":
            aggr_neighbor = neighbor_feature.sum(dim=1)
        elif self.aggr_method == "max":
            aggr_neighbor = neighbor_feature.max(dim=1)
        else:
            raise ValueError("Unknown aggr type, expected sum, max, or mean, but got {}"
                             .format(self.aggr_method))
        # print(aggr_neighbor.shape)
        neighbor_hidden = torch.matmul(aggr_neighbor, self.weight)
        if self.use_bias:
            neighbor_hidden += self.bias

        return neighbor_hidden

    def extra_repr(self):
        return 'in_features={}, out_features={}, aggr_method={}'.format(
            self.input_dim, self.output_dim, self.aggr_method)
    

class SageGCN(nn.Module):
    def __init__(self, input_dim, hidden_dim,
                 activation=F.gelu,
                 aggr_neighbor_method="mean",
                 aggr_hidden_method="sum"):
        """SageGCN layer definition
        # firstworking with mean and concat
        Args:
            input_dim: the dimension of the input feature
            hidden_dim: dimension of hidden layer features,
                When aggr_hidden_method=sum, the output dimension is hidden_dim
                When aggr_hidden_method=concat, the output dimension is hidden_dim*2
            activation: activation function
            aggr_neighbor_method: neighbor feature aggregation method, ["mean", "sum", "max"]
            aggr_hidden_method: update method of node features, ["sum", "concat"]
        """
        super(SageGCN, self).__init__()
        assert aggr_neighbor_method in ["mean", "sum", "max"]
        assert aggr_hidden_method in ["sum", "concat"]
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.aggr_neighbor_method = aggr_neighbor_method
        self.aggr_hidden_method = aggr_hidden_method
        self.activation = activation
        self.aggregator = NeighborAggregator(input_dim, hidden_dim,
                                             aggr_method=aggr_neighbor_method)
        self.weight = nn.Parameter(torch.Tensor(input_dim, hidden_dim))
        self.reset_parameters()
    
    def reset_parameters(self):
        init.kaiming_uniform_(self.weight)

    def forward(self, src_node_features, neighbor_node_features):
        neighbor_hidden = self.aggregator(neighbor_node_features)
        self_hidden = torch.matmul(src_node_features, self.weight)
        
        if self.aggr_hidden_method == "sum":
            hidden = self_hidden + neighbor_hidden
        elif self.aggr_hidden_method == "concat":
            hidden = torch.cat([self_hidden, neighbor_hidden], dim=1)
        else:
            raise ValueError("Expected sum or concat, got {}"
                             .format(self.aggr_hidden))
        if self.activation:
            return self.activation(hidden)
        else:
            return hidden

    def extra_repr(self):
        output_dim = self.hidden_dim if self.aggr_hidden_method == "sum" else self.hidden_dim * 2
        return 'in_features={}, out_features={}, aggr_hidden_method={}'.format(
            self.input_dim, output_dim, self.aggr_hidden_method)


class GraphSage(nn.Module):
    def __init__(self, input_dim, hidden_dim,
                 num_neighbors_list):
        super(GraphSage, self).__init__()
        self.input_dim = input_dim #1433
        self.hidden_dim = hidden_dim #[128, 7]
        self.num_neighbors_list = num_neighbors_list #[10, 10]
        self.num_layers = len(num_neighbors_list)  #2
        self.gcn = nn.ModuleList()
        self.gcn.append(SageGCN(input_dim, hidden_dim[0])) # (1433, 128)
        for index in range(0, len(hidden_dim) - 2):
            self.gcn.append(SageGCN(hidden_dim[index], hidden_dim[index+1])) #128, 7
        self.gcn.append(SageGCN(hidden_dim[-2], hidden_dim[-1], activation=None))

    def forward(self, node_features_list):
        hidden = node_features_list
        for l in range(self.num_layers):
            next_hidden = []
            gcn = self.gcn[l]
            for hop in range(self.num_layers - l):
                src_node_features = hidden[hop]
                src_node_num = len(src_node_features)
                neighbor_node_features = hidden[hop + 1] \
                    .view((src_node_num, self.num_neighbors_list[hop], -1))
                h = gcn(src_node_features, neighbor_node_features)
                next_hidden.append(h)
            hidden = next_hidden
        return hidden[0]

    def extra_repr(self):
        return 'in_features={}, num_neighbors_list={}'.format(
            self.input_dim, self.num_neighbors_list
        )
#### graphsage        


# based on https://github.com/lyakaap/NetVLAD-pytorch/blob/master/netvlad.py
class NetVLAD(nn.Module):
    """NetVLAD layer implementation"""

    def __init__(self, num_clusters=64, dim=128, 
                 normalize_input=True, vladv2=False):
        """
        Args:
            num_clusters : int
                The number of clusters
            dim : int
                Dimension of descriptors
            alpha : float
                Parameter of initialization. Larger value is harder assignment.
            normalize_input : bool
                If true, descriptor-wise L2 normalization is applied to input.
            vladv2 : bool
                If true, use vladv2 otherwise use vladv1
        """
        super(NetVLAD, self).__init__()
        self.num_clusters = num_clusters
        self.dim = dim
        self.alpha = 0
        self.vladv2 = vladv2
        self.normalize_input = normalize_input
        self.conv = nn.Conv2d(dim, num_clusters, kernel_size=(1, 1), bias=vladv2)
        self.graph = nn.Conv1d(128, 64, kernel_size=3, bias=vladv2)
        self.centroids = nn.Parameter(torch.rand(num_clusters, dim))
        
        self.input_dim = 4096
        self.hidden_dim = [4096, 4096]
        self.num_neighbors_list = 5
        self.num_layers = 2
        
        self.gcn = nn.ModuleList()
        self.gcn.append(SageGCN(self.input_dim, self.hidden_dim[0])) # (1433, 128)
        for index in range(0, len(self.hidden_dim) - 2):
            self.gcn.append(SageGCN(self.hidden_dim[index], self.hidden_dim[index+1])) #128, 7
        self.gcn.append(SageGCN(self.hidden_dim[-2], self.hidden_dim[-1], activation=None))
        

    def init_params(self, clsts, traindescs):
        #TODO replace numpy ops with pytorch ops
        if self.vladv2 == False:
            clstsAssign = clsts / np.linalg.norm(clsts, axis=1, keepdims=True)
            dots = np.dot(clstsAssign, traindescs.T)
            dots.sort(0)
            dots = dots[::-1, :] # sort, descending

            self.alpha = (-np.log(0.01) / np.mean(dots[0,:] - dots[1,:])).item()
            self.centroids = nn.Parameter(torch.from_numpy(clsts))
            self.conv.weight = nn.Parameter(torch.from_numpy(self.alpha*clstsAssign).unsqueeze(2).unsqueeze(3))
            self.conv.bias = None
            self.graph.weight = nn.Parameter(torch.from_numpy(self.alpha*clstsAssign[:,:128]).unsqueeze(2))
            self.graph.bias = None
        else:
            knn = NearestNeighbors(n_jobs=-1) #TODO faiss?
            knn.fit(traindescs)
            del traindescs
            dsSq = np.square(knn.kneighbors(clsts, 2)[1])
            del knn
            self.alpha = (-np.log(0.01) / np.mean(dsSq[:,1] - dsSq[:,0])).item()
            self.centroids = nn.Parameter(torch.from_numpy(clsts))
            del clsts, dsSq

            self.conv.weight = nn.Parameter(
                (2.0 * self.alpha * self.centroids).unsqueeze(-1).unsqueeze(-1)
            )
            self.conv.bias = nn.Parameter(
                - self.alpha * self.centroids.norm(dim=1)
            )
    
    def forward(self, x):
        N, C = x.shape[:2]  #8,512,30,40

        if self.normalize_input:
            x = F.normalize(x, p=2, dim=1)  # across descriptor dim

        # soft-assignment
        soft_assign = self.conv(x).view(N, self.num_clusters, -1) #8,64,1200 : (8,64,30,40).view()
        soft_assign = F.softmax(soft_assign, dim=1) #8,64,1200

        x_flatten = x.view(N, C, -1) #8,512,1200
        
        # calculate residuals to each clusters
        vlad = torch.zeros([N, self.num_clusters, C], dtype=x.dtype, layout=x.layout, device=x.device) 
        #vlad: 8, 64,512
        for C in range(self.num_clusters): # slower than non-looped, but lower memory usage 
            #x_flatten.unsqueeze(0).permute(1, 0, 2, 3) = 8,1,512,1200
            #self.centroids[C:C+1, :] : [1,512]
            #self.centroids[C:C+1, :].expand(x_flatten.size(-1), -1, -1) : [1200,1,512]
            # self.centroids[C:C+1, :].expand(x_flatten.size(-1), -1, -1).permute(1, 2, 0).unsqueeze(0) : 1,1,512,1200
            residual = x_flatten.unsqueeze(0).permute(1, 0, 2, 3) - \
                    self.centroids[C:C+1, :].expand(x_flatten.size(-1), -1, -1).permute(1, 2, 0).unsqueeze(0)
            residual *= soft_assign[:,C:C+1,:].unsqueeze(2)
          #  soft_assign[:,C:C+1,:] = [8,1,1200]
          #  soft_assign[:,C:C+1,:].unsqueeze(2) = [8,1,1,1200]
            vlad[:,C:C+1,:] = residual.sum(dim=-1)
            #residual : 8,1,512,1200
            #residual.sum(dim=-1 : [8,1,512])
            #vlad: 8,64,512
        vlad = F.normalize(vlad, p=2, dim=2)  # intra-normalization
        vlad = vlad.view(x.size(0), -1)  # flatten
        vlad = F.normalize(vlad, p=2, dim=1)  # L2 normalize

        return vlad
        
    def forward(self, x):
        #N, C = x.shape[:2]  #8,512,30,40
        N, C, H, W = x.shape
        bb_x = [[0,0,W,H], [0, 0, int(W/3),H], [0, 0, W,int(H/3)], [int(2*W/3), 0, W,H], [0, int(2*H/3), W,H]] # [int(W/4), int(H/4), int(3*W/4),int(3*H/4)]
        vlad = torch.zeros([N, self.num_clusters, C], dtype=x.dtype, layout=x.layout, device=x.device) 
    #   neighborsFeat = torch.zeros([N, 1, self.num_clusters*C], dtype=x.dtype, layout=x.layout, device=x.device) 

        node_features_list = []
        
        for i in range(len(bb_x)):
            
            x_cropped = x[:, : ,bb_x[i][1]:bb_x[i][3], bb_x[i][0]:bb_x[i][2]]
            
            if self.normalize_input:
                x_cropped = F.normalize(x_cropped, p=2, dim=1)  # across descriptor dim
                                
            # soft-assignment
            soft_assign = self.conv(x_cropped).view(N, self.num_clusters, -1) #8,64,1200 : (8,64,30,40).view()
            soft_assign = F.softmax(soft_assign, dim=1) #8,64,1200

            x_flatten = x_cropped.reshape(N, C, -1) #8,512,1200
            
            # calculate residuals to each clusters
            vlad_cropped = torch.zeros([N, self.num_clusters, C], dtype=x_cropped.dtype, layout=x_cropped.layout, device=x_cropped.device) 
            #vlad: 8, 64,512
            for Cc in range(self.num_clusters): # slower than non-looped, but lower memory usage 
                #x_flatten.unsqueeze(0).permute(1, 0, 2, 3) = 8,1,512,1200
                #self.centroids[C:C+1, :] : [1,512]
                #self.centroids[C:C+1, :].expand(x_flatten.size(-1), -1, -1) : [1200,1,512]
                # self.centroids[C:C+1, :].expand(x_flatten.size(-1), -1, -1).permute(1, 2, 0).unsqueeze(0) : 1,1,512,1200
                residual = x_flatten.unsqueeze(0).permute(1, 0, 2, 3) - \
                        self.centroids[Cc:Cc+1, :].expand(x_flatten.size(-1), -1, -1).permute(1, 2, 0).unsqueeze(0)
                residual *= soft_assign[:,Cc:Cc+1,:].unsqueeze(2)
            #  soft_assign[:,C:C+1,:] = [8,1,1200]
            #  soft_assign[:,C:C+1,:].unsqueeze(2) = [8,1,1,1200]
                vlad_cropped[:,Cc:Cc+1,:] = residual.sum(dim=-1)
                #residual : 8,1,512,1200
                #residual.sum(dim=-1 : [8,1,512])
                #vlad: 8,64,512
            #vlad1 = torch.concat([vlad, vlad_cropped],1)
            #vlad = self.graph(vlad1)
            #vlad = torch.add(vlad_cropped,vlad)/2
        
            vlad = F.normalize(vlad_cropped, p=2, dim=2)  # intra-normalization [8,64,512]
            vlad_cropped = []
            vlad = vlad.view(x.size(0), -1)  # flatten [8,32768]
            vlad = F.normalize(vlad, p=2, dim=1)  # L2 normalize
            vlad = vlad.view(-1,4096)
            if (i == 0):
                node_features_list = vlad #[8,4096]
            elif (i == 1):
                neighborsFeat = vlad.unsqueeze(1)
            else:
                neighborsFeat = torch.concat([neighborsFeat, vlad.unsqueeze(1)],1)   
        # print("hello") 
        x_flatten = []
        vlad = []
       
        ## Graphsage
        gcn = self.gcn[0]
        # print(node_features_list.shape)
        # print(neighborsFeat.view(-1,4096).shape)
        vlad = gcn(node_features_list, neighborsFeat)
       # neighborsFeat = []
        vlad = torch.add(node_features_list,vlad)
        
        return vlad.view(-1,32768)



        # for l in range(self.num_layers):
        #     next_hidden = []
        #     gcn = self.gcn[l]
        #     for hop in range(self.num_layers - 1):
        #         src_node_features = hidden[hop]
        #         src_node_num = len(src_node_features)
        #         neighbor_node_features = hidden[hop + 1] \
        #             .view((src_node_num, self.num_neighbors_list, -1))
        #         h = gcn(src_node_features, neighbor_node_features)
        #         next_hidden.append(h)
        #     hidden = next_hidden  
            
        # return hidden[0].view(1,-1)

# class NetVLAD(nn.Module):
#     """NetVLAD layer implementation"""

#     def __init__(self, num_clusters=64, dim=128,
#                  normalize_input=True, vladv2=False, use_faiss=True):
#         """
#         Args:
#             num_clusters : int
#                 The number of clusters
#             dim : int
#                 Dimension of descriptors
#             normalize_input : bool
#                 If true, descriptor-wise L2 normalization is applied to input.
#             vladv2 : bool
#                 If true, use vladv2 otherwise use vladv1
#         """
#         super().__init__()
#         self.num_clusters = num_clusters
#         self.dim = dim
#         self.alpha = 0
#         self.vladv2 = vladv2
#         self.normalize_input = normalize_input
#         self.conv = nn.Conv2d(dim, num_clusters, kernel_size=(1, 1), bias=vladv2)
#         # noinspection PyArgumentList
#         self.centroids = nn.Parameter(torch.rand(num_clusters, dim))
#         self.use_faiss = use_faiss

#     def init_params(self, clsts, traindescs):
#         if not self.vladv2:
#             clstsAssign = clsts / np.linalg.norm(clsts, axis=1, keepdims=True)
#             dots = np.dot(clstsAssign, traindescs.T)
#             dots.sort(0)
#             dots = dots[::-1, :]  # sort, descending

#             self.alpha = (-np.log(0.01) / np.mean(dots[0, :] - dots[1, :])).item()
#             # noinspection PyArgumentList
#             self.centroids = nn.Parameter(torch.from_numpy(clsts))
#             # noinspection PyArgumentList
#             self.conv.weight = nn.Parameter(torch.from_numpy(self.alpha * clstsAssign).unsqueeze(2).unsqueeze(3))
#             self.conv.bias = None
#         else:
#             if not self.use_faiss:
#                 knn = NearestNeighbors(n_jobs=-1)
#                 knn.fit(traindescs)
#                 del traindescs
#                 ds_sq = np.square(knn.kneighbors(clsts, 2)[1])
#                 del knn
#             else:
#                 index = faiss.IndexFlatL2(traindescs.shape[1])
#                 # noinspection PyArgumentList
#                 index.add(traindescs)
#                 del traindescs
#                 # noinspection PyArgumentList
#                 ds_sq = np.square(index.search(clsts, 2)[1])
#                 del index

#             self.alpha = (-np.log(0.01) / np.mean(ds_sq[:, 1] - ds_sq[:, 0])).item()
#             # noinspection PyArgumentList
#             self.centroids = nn.Parameter(torch.from_numpy(clsts))
#             del clsts, ds_sq

#             # noinspection PyArgumentList
#             self.conv.weight = nn.Parameter(
#                 (2.0 * self.alpha * self.centroids).unsqueeze(-1).unsqueeze(-1)
#             )
#             # noinspection PyArgumentList
#             self.conv.bias = nn.Parameter(
#                 - self.alpha * self.centroids.norm(dim=1)
#             )

#     def forward(self, x):
#         N, C = x.shape[:2]

#         if self.normalize_input:
#             x = F.normalize(x, p=2, dim=1)  # across descriptor dim

#         # soft-assignment
#         soft_assign = self.conv(x).view(N, self.num_clusters, -1)
#         soft_assign = F.softmax(soft_assign, dim=1)

#         x_flatten = x.view(N, C, -1)

#         # calculate residuals to each clusters
#         vlad = torch.zeros([N, self.num_clusters, C], dtype=x.dtype, layout=x.layout, device=x.device)
#         for C in range(self.num_clusters):  # slower than non-looped, but lower memory usage
#             residual = x_flatten.unsqueeze(0).permute(1, 0, 2, 3) - \
#                 self.centroids[C:C + 1, :].expand(x_flatten.size(-1), -1, -1).permute(1, 2, 0).unsqueeze(0)
#             residual *= soft_assign[:, C:C + 1, :].unsqueeze(2)
#             vlad[:, C:C + 1, :] = residual.sum(dim=-1)

#         vlad = F.normalize(vlad, p=2, dim=2)  # intra-normalization
#         vlad = vlad.view(x.size(0), -1)  # flatten
#         vlad = F.normalize(vlad, p=2, dim=1)  # L2 normalize

#         return vlad
