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

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors
import faiss
import numpy as np
# graphsage
import torch.nn.init as init

# Semantic Segmentation
from torchvision import transforms
# import espnet as net
from .espnet import *
from torchvision.ops import masks_to_boxes
class NetVLAD(nn.Module):
    """NetVLAD layer implementation"""

    def __init__(self, num_clusters=16, dim=128,
                 normalize_input=True, vladv2=False, use_faiss=True):
        """
        Args:
            num_clusters : int
                The number of clusters
            dim : int
                Dimension of descriptors
            normalize_input : bool
                If true, descriptor-wise L2 normalization is applied to input.
            vladv2 : bool
                If true, use vladv2 otherwise use vladv1
        """
        super().__init__()
        self.num_clusters = num_clusters
        self.dim = dim
        self.alpha = 0
        self.vladv2 = vladv2
        self.normalize_input = normalize_input
        self.conv = nn.Conv2d(dim, num_clusters, kernel_size=(1, 1), bias=vladv2)
        # noinspection PyArgumentList
        self.centroids = nn.Parameter(torch.rand(num_clusters, dim))
        self.use_faiss = use_faiss

    def init_params(self, clsts, traindescs):
        if not self.vladv2:
            clstsAssign = clsts / np.linalg.norm(clsts, axis=1, keepdims=True)
            dots = np.dot(clstsAssign, traindescs.T)
            dots.sort(0)
            dots = dots[::-1, :]  # sort, descending

            self.alpha = (-np.log(0.01) / np.mean(dots[0, :] - dots[1, :])).item()
            # noinspection PyArgumentList
            self.centroids = nn.Parameter(torch.from_numpy(clsts))
            # noinspection PyArgumentList
            self.conv.weight = nn.Parameter(torch.from_numpy(self.alpha * clstsAssign).unsqueeze(2).unsqueeze(3))
            self.conv.bias = None
        else:
            if not self.use_faiss:
                knn = NearestNeighbors(n_jobs=-1)
                knn.fit(traindescs)
                del traindescs
                ds_sq = np.square(knn.kneighbors(clsts, 2)[1])
                del knn
            else:
                index = faiss.IndexFlatL2(traindescs.shape[1])
                # noinspection PyArgumentList
                index.add(traindescs)
                del traindescs
                # noinspection PyArgumentList
                ds_sq = np.square(index.search(clsts, 2)[1])
                del index

            self.alpha = (-np.log(0.01) / np.mean(ds_sq[:, 1] - ds_sq[:, 0])).item()
            # noinspection PyArgumentList
            self.centroids = nn.Parameter(torch.from_numpy(clsts))
            del clsts, ds_sq

            # noinspection PyArgumentList
            self.conv.weight = nn.Parameter(
                (2.0 * self.alpha * self.centroids).unsqueeze(-1).unsqueeze(-1)
            )
            # noinspection PyArgumentList
            self.conv.bias = nn.Parameter(
                - self.alpha * self.centroids.norm(dim=1)
            )

    def forward(self, x):
        N, C = x.shape[:2]

        if self.normalize_input:
            x = F.normalize(x, p=2, dim=1)  # across descriptor dim

        # soft-assignment
        soft_assign = self.conv(x).view(N, self.num_clusters, -1)
        soft_assign = F.softmax(soft_assign, dim=1)
        x_flatten = x.view(N, C, -1)

        residual = x_flatten.expand(self.num_clusters, -1, -1, -1).permute(1, 0, 2, 3) - \
            self.centroids.expand(x_flatten.size(-1), -1, -1).permute(1, 2, 0).unsqueeze(0)
        residual *= soft_assign.unsqueeze(2)
        vlad = residual.sum(dim=-1)

        return vlad

class EmbedNet(nn.Module):
    def __init__(self, base_model, net_vlad):
        super(EmbedNet, self).__init__()
        self.base_model = base_model
        self.net_vlad = net_vlad

    def _init_params(self):
        self.base_model._init_params()
        self.net_vlad._init_params()

    def forward(self, x):
        # pool_x, x = self.base_model(x)
        x = self.base_model(x)
        
        vlad_x = self.net_vlad(x)

        # [IMPORTANT] normalize
        vlad_x = F.normalize(vlad_x, p=2, dim=2)  # intra-normalization
        vlad_x = vlad_x.view(x.size(0), -1)  # flatten
        vlad_x = F.normalize(vlad_x, p=2, dim=1)  # L2 normalize

        return vlad_x

class EmbedNetPCA(nn.Module):
    def __init__(self, base_model, net_vlad, dim=4096):
        super(EmbedNetPCA, self).__init__()
        self.base_model = base_model
        self.net_vlad = net_vlad
        self.pca_layer = nn.Conv2d(net_vlad.centroids.shape[0]*net_vlad.centroids.shape[1], dim, 1, stride=1, padding=0)

    def _init_params(self):
        self.base_model._init_params()
        self.net_vlad._init_params()

    def forward(self, x):
        # _, x = self.base_model(x)
        x = self.base_model(x)
        vlad_x = self.net_vlad(x)

        # [IMPORTANT] normalize
        vlad_x = F.normalize(vlad_x, p=2, dim=2)  # intra-normalization
        vlad_x = vlad_x.view(x.size(0), -1)  # flatten
        vlad_x = F.normalize(vlad_x, p=2, dim=1)  # L2 normalize

        # reduction
        N, D = vlad_x.size()
        vlad_x = vlad_x.view(N, D, 1, 1)
        vlad_x = self.pca_layer(vlad_x).view(N, -1)
        vlad_x = F.normalize(vlad_x, p=2, dim=-1)  # L2 normalize

        return vlad_x
    
#### graphsage     
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
            aggr_neighbor = torch.amax(neighbor_feature, 1)
        else:
            raise ValueError("Unknown aggr type, expected sum, max, or mean, but got {}"
                             .format(self.aggr_method))
        # print(aggr_neighbor.shape,self.weight.shape)
        neighbor_hidden = torch.matmul(aggr_neighbor, self.weight)
        if self.use_bias:
            neighbor_hidden += self.bias

        return neighbor_hidden

    def extra_repr(self):
        return 'in_features={}, out_features={}, aggr_method={}'.format(
            self.input_dim, self.output_dim, self.aggr_method)
    
#F.pre PReLU
# prelu
class SageGCN(nn.Module):
    def __init__(self, input_dim, hidden_dim,
                 activation=F.gelu,
                 aggr_neighbor_method="sum",
                 aggr_hidden_method="concat"):
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
        
        # print('src_node_features', neighbor_node_features.shape, src_node_features.shape, self.weight.shape)
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
        # code.interact(local=locals())
        subfeat_size = int(hidden[0].shape[1]/self.input_dim) #8
        gcndim = int(self.input_dim) 
        
        # print('subfeat_size ', subfeat_size)
        # print('  l  ', ' hop  ', '  src_node_features  ', '  neighbor_node_features  ', '  h  ', '    ')

        for l in range(self.num_layers):
            next_hidden = []
            gcn = self.gcn[l]
            for hop in range(self.num_layers - l):
                src_node_features = hidden[hop] #torch.Size([4, 32768])
                src_node_num = len(src_node_features) # 4
                # print('neighbor_node_features ', hidden[hop + 1].shape  ,' / ',  src_node_num, self.num_neighbors_list[hop], '-1')
                neighbor_node_features = hidden[hop + 1] \
                    .view((src_node_num, self.num_neighbors_list[hop], -1))
                # splitting the i/p 32768 = 8 x 4096
                # src_node_features N x 4096
                # neighbor_node_features N x 5 x 4096       
                for j in range(subfeat_size):    # 8X4096 = 32768 and 8xNx4096
                    h_x = gcn(src_node_features[:,j*gcndim:j*gcndim+gcndim], neighbor_node_features[:,:,j*gcndim:j*gcndim+gcndim])
                    # neighborsFeat = []
                    if (j==0):
                        h = h_x; # 4 x 4096
                    else:
                        h = torch.concat([h, h_x],1) 
                        
                # print("hop", hop,'  ',  h.shape)
                next_hidden.append(h)
            hidden = next_hidden
        # print("hidden", ' ',  hidden[0].shape)    
        return hidden[0]

    def extra_repr(self):
        return 'in_features={}, num_neighbors_list={}'.format(
            self.input_dim, self.num_neighbors_list
        )


class GraphVLAD(nn.Module):
    def __init__(self, base_model, net_vlad, esp_net):
        super(GraphVLAD, self).__init__()
        self.base_model = base_model
        self.net_vlad = net_vlad
        self.Espnet = esp_net
        
        #graph
        self.input_dim = 4096 # 16384# 8192
        self.hidden_dim = [2048,4096]#[8192, 8192]
        self.num_neighbors_list = [5]#,2]
        
        self.graph = GraphSage(input_dim=self.input_dim, hidden_dim=self.hidden_dim,
                  num_neighbors_list=self.num_neighbors_list)

    def _init_params(self):
        self.base_model._init_params()
        self.net_vlad._init_params()

    def forward(self, x):
        # fixing for Tokyo
        sizeH = x.shape[2]
        sizeW = x.shape[3]
        if sizeH%2 != 0:
            x = F.pad(input=x, pad=(0,0,1,2), mode='constant', value=0)
        if sizeW%2 != 0:
            x = F.pad(input=x, pad=(1,2), mode='constant', value=0)
        # Create Segmentation    
        with torch.no_grad():
            b_out = self.Espnet(x)
        mask = b_out.max(1)[1] 
        for jj in range(len(mask)):  #batch processing
            single_label_mask = mask[jj]
            obj_ids, obj_i = single_label_mask.unique(return_counts=True)
            obj_ids = obj_ids[1:] 
            obj_i = obj_i[1:]
            masks = single_label_mask == obj_ids[:, None, None]
            boxes_t = masks_to_boxes(masks.to(torch.float32))
            # Sort the boxes                
            rr_boxes = torch.argsort(torch.argsort(obj_i,descending=True)) # (decending order)
            boxes = boxes_t/16
    
        _, _, H, W = x.shape
        patch_mask = torch.zeros((H, W)).cuda()
        # VGG 
        x = self.base_model(x)  

        N, C, H, W = x.shape

        bb_x = [[int(W/4), int(H/4), int(3*W/4),int(3*H/4)],
                [0, 0, int(W/3),H], 
                [0, 0, W,int(H/3)], 
                [int(2*W/3), 0, W,H], 
                [0, int(2*H/3), W,H]]

        NB = 5
        graph_nodes = torch.zeros(N,NB,C,H,W).cuda()
        rsizet = transforms.Resize((H,W)) #H W

        for Nx in range(N):    
            img_nodes = []
            for idx in range(len(boxes)):
                for b_idx in range(len(rr_boxes)):
                    if idx == rr_boxes[b_idx] and obj_i[b_idx] > 5000 and len(img_nodes) < NB:     
                        patch_mask = patch_mask*0
                        patch_mask[single_label_mask == obj_ids[b_idx]] = 1
                        patch_maskr = rsizet(patch_mask.unsqueeze(0))
                        patch_maskr = patch_maskr.squeeze(0)
                        boxesd = boxes.to(torch.long)
                        x_min,y_min,x_max,y_max = boxesd[b_idx]
                        c_img = x[Nx][:, y_min:y_max,x_min:x_max]
                        resultant = rsizet(c_img)
                        img_nodes.append(resultant.unsqueeze(0))
                        break                    
            if len(img_nodes) < NB:
                for i in range(len(bb_x)-len(img_nodes)):
                    x_cropped =  x[Nx][: ,bb_x[i][1]:bb_x[i][3], bb_x[i][0]:bb_x[i][2]]
                    img_nodes.append(rsizet(x_cropped.unsqueeze(0)))
            aa = torch.stack(img_nodes,1)
            graph_nodes[Nx] = aa[0]
        
                        
        node_features_list = []
        neighborsFeat = []
        x_cropped = graph_nodes.view(NB,N,C,H,W)
        x_cropped = torch.cat((graph_nodes.view(NB,N,C,H,W), x.unsqueeze(0)))
         
        for i in range(NB+1):
            vlad_x = self.net_vlad(x_cropped[i])
            
            # [IMPORTANT] normalize
            vlad_x = F.normalize(vlad_x, p=2, dim=2)  # intra-normalization
            vlad_x = vlad_x.view(x.size(0), -1)  # flatten
            vlad_x = F.normalize(vlad_x, p=2, dim=1)  # L2 normalize

            neighborsFeat.append(vlad_x)

        node_features_list.append(neighborsFeat[NB])
        node_features_list.append(torch.concat(neighborsFeat[0:NB],0))
        neighborsFeat = []
        
        ## Graphsage
        gvlad = self.graph(node_features_list)
        gvlad = torch.add(gvlad,vlad_x)        
        return gvlad.view(-1,vlad_x.shape[1])
    
class GraphVLADPCA(nn.Module):
    def __init__(self, base_model, net_vlad, dim=4096):
        super(GraphVLADPCA, self).__init__()
        self.base_model = base_model
        self.net_vlad = net_vlad
        self.pca_layer = nn.Conv2d(net_vlad.centroids.shape[0]*net_vlad.centroids.shape[1], dim, 1, stride=1, padding=0)

        # Semantic Segmentation
        self.classes = 20
        self.p = 2
        self.q = 8
        self.encoderFile = "/home/leo/usman_ws/datasets/espnet-encoder/espnet_p_2_q_8.pth"
        self.Espnet = ESPNet(classes=self.classes, p=self.p, q = self.q, encoderFile=self.encoderFile)  # Net.
        
        #graph
        self.input_dim = 4096 # 16384# 8192
        self.hidden_dim = [2048,4096]#[8192, 8192]
        self.num_neighbors_list = [5]#,2]
        
        self.graph = GraphSage(input_dim=self.input_dim, hidden_dim=self.hidden_dim,
                  num_neighbors_list=self.num_neighbors_list)

    def _init_params(self):
        self.base_model._init_params()
        self.net_vlad._init_params()

    def forward(self, x):
        
        # fixing for Tokyo
        sizeH = x.shape[2]
        sizeW = x.shape[3]
        # print("raw: ", x.shape)
        
        # for Tokyo 247 Test
        
        if sizeH%2 != 0:
            x = F.pad(input=x, pad=(0,0,1,2), mode='constant', value=0)
        if sizeW%2 != 0:
            x = F.pad(input=x, pad=(1,2), mode='constant', value=0)

        # print("padded:", x.shape)
        with torch.no_grad():
            b_out = self.Espnet(x)
        # b_out = self.Espnet(x)
        mask = b_out.max(1)[1]   #torch.Size([36, 480, 640])
        
        for jj in range(len(mask)):  #batch processing
            
            # img_orig = to_tensor(Image.open(image_list[jj]).convert('RGB'))
            # _, H, W = img_orig.shape

            # bb_x = [[0, 0, round(2*W/3), round(2*H/3)],  [round(W/3),  0,  W, round(2*H/3)], [0, round(H/3), round(2*W/3), H], [round(W/3), round(H/3), W, H]]
                
            # patch_mask = torch.zeros((H, W))
            
            # rsizet = transforms.Resize((427, 320)) #H W
            # rsizet = transforms.Resize((round(2*W/3), round(2*H/3))) #H W
            

             #patch_mask, patch_mask, patch_mask, patch_mask]

            # single_label_mask = relabel_merge(mask[jj])    # single image mask
            # all the labels to single slides
            single_label_mask = mask[jj]
            
            # obj_ids = torch.unique(single_label_mask)
            obj_ids, obj_i = single_label_mask.unique(return_counts=True)
            obj_ids = obj_ids[1:] 
            obj_i = obj_i[1:]
            #torch.Size([19])
            
            masks = single_label_mask == obj_ids[:, None, None]
            boxes_t = masks_to_boxes(masks.to(torch.float32))
            # print ("boxes-lenght:", len(boxes_t))

            # Sort the boxes                
            # rr = ((bb_x[:, 2])-(bb_x[:, 0]))*((bb_x[:, 3])-(bb_x[:, 1]))
            # rr_boxes = torch.argsort(rr,descending=True) # (decending order)
            
            rr_boxes = torch.argsort(torch.argsort(obj_i,descending=True)) # (decending order)

            
            # rr_boxes = torch.argsort(rr) # (decending order)

            # boxes = (boxes_t/16).cpu().numpy().astype(int)
            
            boxes = boxes_t/16
        
        
        # for jj in range(len(mask)):  #batch processing
        
        #     single_label_mask = relabel_merge(mask[jj])    # single image mask
        #     # single_label_mask = mask[jj]
        #     obj_ids = torch.unique(single_label_mask)
        #     obj_ids = obj_ids[1:]      #torch.Size([19])
        #     masks = single_label_mask == obj_ids[:, None, None]
        #     boxes = masks_to_boxes(masks.to(torch.float32))
        #     # boxes = masks_to_boxes(masks.to(torch.float32))/16
        #     boxes_s = (boxes/16).cpu().numpy().astype(int)
        #     #append boxes
        #     print (len(boxes_s))
        #     code.interact(local=locals())

            
        _, _, H, W = x.shape
        # H = sizeH
        # W = sizeW
        patch_mask = torch.zeros((H, W)).cuda()
        
        # VGG 
        x = self.base_model(x)   
        
        N, C, H, W = x.shape

        
        # img_orig = to_tensor(Image.open(image_list[jj]).convert('RGB'))
        # _, H, W = img_orig.shape

        bb_x = [[int(W/4), int(H/4), int(3*W/4),int(3*H/4)],
                [0, 0, int(W/3),H], 
                [0, 0, W,int(H/3)], 
                [int(2*W/3), 0, W,H], 
                [0, int(2*H/3), W,H]]

        # bb_x = [[0, 0, round(2*W/3), round(2*H/3)],  [round(W/3),  0,  W, round(2*H/3)], [0, round(H/3), round(2*W/3), H], [round(W/3), round(H/3), W, H]]
            
        # patch_mask = torch.zeros((H, W))
        NB = 5
        
        graph_nodes = torch.zeros(N,NB,C,H,W).cuda()
        rsizet = transforms.Resize((H,W)) #H W

        # rsizet = transforms.Resize((427, 320)) #H W
        # rsizet = transforms.Resize((round(2*W/3), round(2*H/3))) #H W
        
        for Nx in range(N):    
            # img_stk = x[Nx].unsqueeze(0)
            img_nodes = []
            # print(Nx)
            for idx in range(len(boxes)):
                for b_idx in range(len(rr_boxes)):
                    # print(idx, " ", b_idx)
                    # code.interact(local=locals())

                    if idx == rr_boxes[b_idx] and obj_i[b_idx] > 5000 and len(img_nodes) < NB:
                        # print("found match")
                        # print(idx, " ", b_idx)
                        # print (img_nodes.shape)
                        
                        patch_mask = patch_mask*0

                        # label obj_ids[rr_boxes[b_idx]]
                        patch_mask[single_label_mask == obj_ids[b_idx]] = 1
                        # box boxes[rr_boxes[b_idx]]

                        # patch_mask = patch_mask.unsqueeze(0)
                        patch_maskr = rsizet(patch_mask.unsqueeze(0))
                        
                        patch_maskr = patch_maskr.squeeze(0)

                        boxesd = boxes.to(torch.long)
                        x_min,y_min,x_max,y_max = boxesd[b_idx]
                    
                        # zero_img = patch_maskr[y_min:y_max,x_min:x_max]
                    
                        # imgg = img[0].permute(1, 2, 0).numpy().astype(int)
                        c_img = x[Nx][:, y_min:y_max,x_min:x_max]
                        
                        # increase dimension
                        # mmask = torch.stack((zero_img,)*512, axis=0)
                        

                        # Multiply arrays
                        # code.interact(local=locals())
                        # resultant = rsizet(c_img*mmask)
                        resultant = rsizet(c_img)
 
                        img_nodes.append(resultant.unsqueeze(0))
                        
                        
                        # img_nodes = torch.stack((img_nodes,resultant.unsqueeze(0)), 0)
                        # code.interact(local=locals())
                        # img_nodes.append(resultant.unsqueeze(0))
                        

                        # imgg = torch.permute(resultant, (1, 2, 0)).cpu().numpy()[0]
                        # aa = img_orig.numpy()
                        # imgg = to_image(aa)
                        
                        # cv2.imwrite(args.savedir + os.sep + 'img_'+str(idx)+'_' + name.replace(args.img_extn, 'png'), aa)
                        # save_image(resultant, args.savedir + os.sep + 'img_'+str(idx)+'_' + name.replace(args.img_extn, 'png'))
                        break                    
            
            # check the size
            # print("first: ", len(img_nodes))
            # code.interact(local=locals())
            if len(img_nodes) < NB:
                for i in range(len(bb_x)-len(img_nodes)):
                    x_cropped =  x[Nx][: ,bb_x[i][1]:bb_x[i][3], bb_x[i][0]:bb_x[i][2]]
                    img_nodes.append(rsizet(x_cropped.unsqueeze(0)))
                    
                
        
            aa = torch.stack(img_nodes,1)
            # code.interact(local=locals())
            graph_nodes[Nx] = aa[0]
            # graph_nodes.append(torch.stack(img_nodes,1))
            # print("total: ", len(img_nodes))
        # code.interact(local=locals())
        
        
        node_features_list = []
        neighborsFeat = []

        x_cropped = graph_nodes.view(NB,N,C,H,W)
        # xx = x.unsqueeze(0)
        # Append root node
        # print(x_cropped.shape)
        # print(x.unsqueeze(0).shape)
        # code.interact(local=locals())
        x_cropped = torch.cat((graph_nodes.view(NB,N,C,H,W), x.unsqueeze(0)))
        # x_call = x_cropped.append(x.unsqueeze(0))
         
        for i in range(NB+1):
            
            
            vlad_x = self.net_vlad(x_cropped[i])
            
            # [IMPORTANT] normalize
            vlad_x = F.normalize(vlad_x, p=2, dim=2)  # intra-normalization
            vlad_x = vlad_x.view(x.size(0), -1)  # flatten
            vlad_x = F.normalize(vlad_x, p=2, dim=1)  # L2 normalize
            # aa = vlad_x.shape #32, 32768
            #vlad_x = vlad_x.view(-1,8192) # 8192
            # print(i)
            
            neighborsFeat.append(vlad_x)

        #code.interact(local=locals())
        node_features_list.append(neighborsFeat[NB])
        node_features_list.append(torch.concat(neighborsFeat[0:NB],0))
        # code.interact(local=locals())
        
        neighborsFeat = []
        #vlad_x = []
        # vlad_x.shape[1]
        ## Graphsage
        gvlad = self.graph(node_features_list) 
        #torch.Size([4, 32768])
        gvlad = torch.add(gvlad,vlad_x)

        # gvlad = F.normalize(gvlad, p=2, dim=1)  # L2 normalize
        
        gvlad = gvlad.view(-1,vlad_x.shape[1])
        # print(gvlad.shape)
        # reduction
        N, D = gvlad.size()
        gvlad = gvlad.view(N, D, 1, 1)
        gvlad = self.pca_layer(gvlad).view(N, -1)
        gvlad = F.normalize(gvlad, p=2, dim=-1)  # L2 normalize
        
        return gvlad

        