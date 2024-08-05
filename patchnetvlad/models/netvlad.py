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
import torch.nn.init as init
from torchvision import transforms
from torchvision.ops import masks_to_boxes
from .espnet import *
from .visualize import get_color_pallete, save_image
from PIL import Image

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
        if self.aggr_method == "mean":
            aggr_neighbor = neighbor_feature.mean(dim=1)
        elif self.aggr_method == "sum":
            aggr_neighbor = neighbor_feature.sum(dim=1)
        elif self.aggr_method == "max":
            aggr_neighbor = torch.amax(neighbor_feature, 1)
        else:
            raise ValueError("Unknown aggr type, expected sum, max, or mean, but got {}"
                             .format(self.aggr_method))
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
                 aggr_neighbor_method="sum",
                 aggr_hidden_method="concat"):
        """SageGCN layer definition
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
class GraphSage(nn.Module):
    def __init__(self, input_dim, hidden_dim,
                 num_neighbors_list):
        super(GraphSage, self).__init__()
        self.input_dim = input_dim 
        self.hidden_dim = hidden_dim 
        self.num_neighbors_list = num_neighbors_list 
        self.num_layers = len(num_neighbors_list)  
        self.gcn = nn.ModuleList()
        self.gcn.append(SageGCN(input_dim, hidden_dim[0])) 
        for index in range(0, len(hidden_dim) - 2):
            self.gcn.append(SageGCN(hidden_dim[index], hidden_dim[index+1])) 
        self.gcn.append(SageGCN(hidden_dim[-2], hidden_dim[-1], activation=None))
    def forward(self, node_features_list):
        hidden = node_features_list
        subfeat_size = int(hidden[0].shape[1]/self.input_dim) 
        gcndim = int(self.input_dim) 
        for l in range(self.num_layers):
            next_hidden = []
            gcn = self.gcn[l]
            for hop in range(self.num_layers - l):
                src_node_features = hidden[hop] 
                src_node_num = len(src_node_features) 
                neighbor_node_features = hidden[hop + 1] \
                    .view((src_node_num, self.num_neighbors_list[hop], -1))
                for j in range(subfeat_size):    
                    h_x = gcn(src_node_features[:,j*gcndim:j*gcndim+gcndim], neighbor_node_features[:,:,j*gcndim:j*gcndim+gcndim])
                    if (j==0):
                        h = h_x; 
                    else:
                        h = torch.concat([h, h_x],1) 
                next_hidden.append(h)
            hidden = next_hidden
        return hidden[0]
    def extra_repr(self):
        return 'in_features={}, num_neighbors_list={}'.format(
            self.input_dim, self.num_neighbors_list
        )
class NetVLAD(nn.Module):
    """NetVLAD layer implementation"""

    def __init__(self, num_clusters=64, dim=512, alpha=100.0, normalize_input=True):
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
        """
        super(NetVLAD, self).__init__()
        self.num_clusters = num_clusters
        self.dim = dim
        self.alpha = alpha
        self.normalize_input = normalize_input
        self.conv = nn.Conv2d(dim, num_clusters, kernel_size=(1, 1), bias=False)
        self.centroids = nn.Parameter(torch.rand(num_clusters, dim), requires_grad=True)

        self.clsts = None
        self.traindescs = None
        




    def _init_params(self):
        clstsAssign = self.clsts / np.linalg.norm(self.clsts, axis=1, keepdims=True)
        dots = np.dot(clstsAssign, self.traindescs.T)
        dots.sort(0)
        dots = dots[::-1, :] # sort, descending

        self.alpha = (-np.log(0.01) / np.mean(dots[0,:] - dots[1,:])).item()
        self.centroids.data.copy_(torch.from_numpy(self.clsts))
        self.conv.weight.data.copy_(torch.from_numpy(self.alpha*clstsAssign).unsqueeze(2).unsqueeze(3))

    def forward(self, x):
        N, C = x.shape[:2]
        
        if self.normalize_input:
            x = F.normalize(x, p=2, dim=1)  # across descriptor dim

        # soft-assignment
        soft_assign = self.conv(x).view(N, self.num_clusters, -1)
        soft_assign = F.softmax(soft_assign, dim=1)

        x_flatten = x.view(N, C, -1)

        # calculate residuals to each clusters in one loop
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
        pool_x, x = self.base_model(x)
        vlad_x = self.net_vlad(x)
        vlad_x = F.normalize(vlad_x, p=2, dim=2)  
        vlad_x = vlad_x.view(x.size(0), -1)  
        vlad_x = F.normalize(vlad_x, p=2, dim=1)  
        return pool_x, vlad_x
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
        _, x = self.base_model(x)
        vlad_x = self.net_vlad(x)
        vlad_x = F.normalize(vlad_x, p=2, dim=2)  
        vlad_x = vlad_x.view(x.size(0), -1)  
        vlad_x = F.normalize(vlad_x, p=2, dim=1)  
        N, D = vlad_x.size()
        vlad_x = vlad_x.view(N, D, 1, 1)
        vlad_x = self.pca_layer(vlad_x).view(N, -1)
        vlad_x = F.normalize(vlad_x, p=2, dim=-1)  
        return vlad_x
class EmbedRegionNet(nn.Module):
    def __init__(self, base_model, net_vlad, tuple_size=1):
        super(EmbedRegionNet, self).__init__()
        self.base_model = base_model
        self.net_vlad = net_vlad
        self.tuple_size = tuple_size

    def _init_params(self):
        self.base_model._init_params()
        self.net_vlad._init_params()

    def _compute_region_sim(self, feature_A, feature_B):
        # feature_A: B*C*H*W
        # feature_B: (B*(1+neg_num))*C*H*W

        def reshape(x):
            # re-arrange local features for aggregating quarter regions
            N, C, H, W = x.size()
            x = x.view(N, C, 2, int(H/2), 2, int(W/2))
            x = x.permute(0,1,2,4,3,5).contiguous()
            x = x.view(N, C, -1, int(H/2), int(W/2))
            return x

        feature_A = reshape(feature_A)
        feature_B = reshape(feature_B)

        # compute quarter-region features
        def aggregate_quarter(x):
            N, C, B, H, W = x.size()
            x = x.permute(0,2,1,3,4).contiguous()
            x = x.view(-1,C,H,W)
            vlad_x = self.net_vlad(x) # (N*B)*64*512
            _, cluster_num, feat_dim = vlad_x.size()
            vlad_x = vlad_x.view(N,B,cluster_num,feat_dim)
            return vlad_x

        vlad_A_quarter = aggregate_quarter(feature_A)
        vlad_B_quarter = aggregate_quarter(feature_B)

        # compute half-region features
        def quarter_to_half(vlad_x):
            return torch.stack((vlad_x[:,0]+vlad_x[:,1], vlad_x[:,2]+vlad_x[:,3], \
                                vlad_x[:,0]+vlad_x[:,2], vlad_x[:,1]+vlad_x[:,3]), dim=1).contiguous()

        vlad_A_half = quarter_to_half(vlad_A_quarter)
        vlad_B_half = quarter_to_half(vlad_B_quarter)

        # compute global-image features
        def quarter_to_global(vlad_x):
            return vlad_x.sum(1).unsqueeze(1).contiguous()

        vlad_A_global = quarter_to_global(vlad_A_quarter)
        vlad_B_global = quarter_to_global(vlad_B_quarter)

        def norm(vlad_x):
            N, B, C, _ = vlad_x.size()
            vlad_x = F.normalize(vlad_x, p=2, dim=3)  # intra-normalization
            vlad_x = vlad_x.view(N, B, -1)  # flatten
            vlad_x = F.normalize(vlad_x, p=2, dim=2)  # L2 normalize
            return vlad_x

        vlad_A = torch.cat((vlad_A_global, vlad_A_half, vlad_A_quarter), dim=1)
        vlad_B = torch.cat((vlad_B_global, vlad_B_half, vlad_B_quarter), dim=1)
        vlad_A = norm(vlad_A)
        vlad_B = norm(vlad_B)

        _, B, L = vlad_B.size()
        vlad_A = vlad_A.view(self.tuple_size,-1,B,L)
        vlad_B = vlad_B.view(self.tuple_size,-1,B,L)

        score = torch.bmm(vlad_A.expand_as(vlad_B).view(-1,B,L), vlad_B.view(-1,B,L).transpose(1,2))
        score = score.view(self.tuple_size,-1,B,B)

        return score, vlad_A, vlad_B

    def _forward_train(self, x):
        B, C, H, W = x.size()
        x = x.view(self.tuple_size, -1, C, H, W)

        anchors = x[:, 0].unsqueeze(1).contiguous().view(-1,C,H,W) # B*C*H*W
        pairs = x[:, 1:].view(-1,C,H,W) # (B*(1+neg_num))*C*H*W

        return self._compute_region_sim(anchors, pairs)

    def forward(self, x):
        pool_x, x = self.base_model(x)

        if (not self.training):
            vlad_x = self.net_vlad(x)
            # normalize
            vlad_x = F.normalize(vlad_x, p=2, dim=2)  # intra-normalization
            vlad_x = vlad_x.view(x.size(0), -1)  # flatten
            vlad_x = F.normalize(vlad_x, p=2, dim=1)  # L2 normalize
            return pool_x, vlad_x

        return self._forward_train(x)

class applyGNN(nn.Module):
    def __init__(self):
        super(applyGNN, self).__init__()
        self.input_dim = 4096 
        self.hidden_dim = [2048,2048]
        self.num_neighbors_list = [5]
        self.graph = GraphSage(input_dim=self.input_dim, hidden_dim=self.hidden_dim,
                  num_neighbors_list=self.num_neighbors_list)
    def forward(self, x):
        gvlad = self.graph(x)
        return gvlad
class SelectRegions(nn.Module):
    def __init__(self, NB, Mask):
        super(SelectRegions, self).__init__()
        self.NB = NB
        self.mask = Mask
        
    def relabel(self, img):
        """
        This function relabels the predicted labels so that cityscape dataset can process
        :param img: The image array to be relabeled
        :return: The relabeled image array
        """
        ### Road 0 + Sidewalk 1
        img[img == 1] = 1
        img[img == 0] = 1

        ### building 2 + wall 3 + fence 4
        img[img == 2] = 2
        img[img == 3] = 2
        img[img == 4] = 2

        ### vegetation 8 + Terrain 9
        img[img == 9] = 3
        img[img == 8] = 3

        ### Pole 5 + Traffic Light 6 + Traffic Signal 7
        img[img == 7] = 4
        img[img == 6] = 4
        img[img == 5] = 4
        
        ### Sky 10
        img[img == 10] = 5
        
        ## Rider 12 + motorcycle 17 + bicycle 18
        img[img == 18] = 255
        img[img == 17] = 255
        img[img == 12] = 255


        # cars 13 + truck 14 + bus 15 + train 16
        img[img == 16] = 255
        img[img == 15] = 255
        img[img == 14] = 255
        img[img == 13] = 255

        ## Person
        img[img == 11] = 255

        ### Don't need, make these 255
        ## Background
        img[img == 19] = 255

        return img                          
    
    def forward(self, x, base_model, fastscnn): 
        
        ## debug
        # save_image(x[0], 'output-image.png')
        # mask = get_color_pallete(pred_g_merge[0].cpu().numpy())
        # mask.save('output.png')
        sizeH = x.shape[2]
        sizeW = x.shape[3]
        
        # Pad if height or width is odd
        if sizeH % 2 != 0:
            x = F.pad(input=x, pad=(0, 0, 1, 2), mode="constant", value=0)
        if sizeW % 2 != 0:
            x = F.pad(input=x, pad=(1, 2), mode="constant", value=0)

        # Forward pass through fastscnn without gradients
        with torch.no_grad():
            outputs = fastscnn(x)

        # save_image(x[0], 'output-image.png')
        # Forward pass through base_model
        pool_x, x = base_model(x)
        N, C, H, W = x.shape
        
        # Initialize graph nodes tensor
        graph_nodes = torch.zeros(N, self.NB, C, H, W).cuda()
        rsizet = transforms.Resize((H, W))
        
        # Process the output of fastscnn to get predicted labels
        pred_all = torch.argmax(outputs[0], 1)
        # mask = get_color_pallete(pred_all[0].cpu().numpy(), 'citys')
        # mask.save('output.png')
        
        
        # pred_all_c = torch.argmax(outputs[0], 1).squeeze(0).cpu().data.numpy()
        # mask = get_color_pallete(pred_all_c[0], 'citys')
        # mask.save('output.png')
        
        
        pred_all = self.relabel(pred_all)

        # mask = get_color_pallete(pred_all[0].cpu().numpy(), 'citys')
        # mask.save('output-m.png')
        for img_i in range(N):
            all_label_mask = pred_all[img_i]
            labels_all, label_count_all = all_label_mask.unique(return_counts=True)
            
            mask_t = label_count_all >= 10000
            labels = labels_all[mask_t]
            
            # Create masks for each label and convert them to bounding boxes
            masks = all_label_mask == labels[:, None, None]
            regions = masks_to_boxes(masks.to(torch.float32))
            boxes = (regions / 16).to(torch.long)
            all_label_mask = rsizet(all_label_mask.unsqueeze(0)).squeeze(0)

            sub_nodes = []
            pre_l2 = x[img_i]
            
            if self.mask:
                for i, label in enumerate(labels):
                    binary_mask = (all_label_mask == label).float()
                    embed_image = (pre_l2 * binary_mask) + pre_l2
                sub_nodes.append(embed_image.unsqueeze(0))

            # sub_nodes.append(filterd_img.unsqueeze(0))
            # Add more patches by cropping predefined regions if needed
            if len(sub_nodes) < self.NB:
                bb_x = [
                    [0, 0, int(2 * W / 3), H],
                    [int(W / 3), 0, W, H],
                    [0, 0, W, int(2 * H / 3)],
                    [0, int(H / 3), W, H],
                    [int(W / 4), int(H / 4), int(3 * W / 4), int(3 * H / 4)],
                ]
                for i in range(len(bb_x) - len(sub_nodes)):
                    x_nodes = embed_image[:, bb_x[i][1]:bb_x[i][3], bb_x[i][0]:bb_x[i][2]]
                    sub_nodes.append(rsizet(x_nodes.unsqueeze(0)))

            # Stack the cropped patches and store them in graph_nodes
            aa = torch.stack(sub_nodes, 1)
            graph_nodes[img_i] = aa[0]

        # Reshape and concatenate graph_nodes with the original tensor x
        x_nodes = graph_nodes.view(self.NB, N, C, H, W)
        x_nodes = torch.cat((x_nodes, x.unsqueeze(0)))
        
        # Clean up
        del graph_nodes, sub_nodes, pred_all, labels_all, label_count_all, masks, regions, boxes, all_label_mask
        
        return pool_x, x.size(0), x_nodes
class GraphVLAD(nn.Module):
    def __init__(self, base_model, net_vlad, fastscnn, NB):
        super(GraphVLAD, self).__init__()
        self.base_model = base_model
        self.fastscnn = fastscnn
        self.net_vlad = net_vlad
        
        self.NB = NB
        self.mask = True
                
        self.applyGNN = applyGNN()
        self.SelectRegions = SelectRegions(self.NB, self.mask)

    def _init_params(self):
        self.base_model._init_params()
        self.net_vlad._init_params()

    def forward(self, x):
        node_features_list = []
        pool_x, x_size, x_nodes = self.SelectRegions(x, self.base_model, self.fastscnn)
        
        neighborsFeat = []
        for i in range(self.NB + 1):
            vlad_x = self.net_vlad(x_nodes[i])
            vlad_x = F.normalize(vlad_x, p=2, dim=2)
            vlad_x = vlad_x.view(x_size, -1)
            vlad_x = F.normalize(vlad_x, p=2, dim=1)
            neighborsFeat.append(vlad_x)
        
        node_features_list.append(neighborsFeat[self.NB])
        node_features_list.append(torch.cat(neighborsFeat[0:self.NB], 0))
        
        # Clear neighborsFeat to free up memory
        del neighborsFeat
        
        gvlad = self.applyGNN(node_features_list)
        gvlad = torch.add(gvlad, vlad_x)
        gvlad = gvlad.view(-1, vlad_x.shape[1])
        
        # Clear node_features_list to free up memory
        del node_features_list
        
        return pool_x, gvlad
class GraphVLADPCA(nn.Module):
    def __init__(self, base_model, net_vlad, esp_net, dim=4096):
        super(GraphVLADPCA, self).__init__()
        self.base_model = base_model
        self.esp_net = esp_net
        self.net_vlad = net_vlad
        self.SelectRegions = SelectRegions()
        self.applyGNN = applyGNN()
        self.pca_layer = nn.Conv2d(net_vlad.centroids.shape[0]*net_vlad.centroids.shape[1], dim, 1, stride=1, padding=0)
    def _init_params(self):
        self.base_model._init_params()
        self.net_vlad._init_params()
    def forward(self, x):
        node_features_list = []
        neighborsFeat = []
        NB, x_size, x_cropped = self.SelectRegions(x, self.base_model, self.esp_net)
        for i in range(NB+1):
            vlad_x = self.net_vlad(x_cropped[i])
            vlad_x = F.normalize(vlad_x, p=2, dim=2)  
            vlad_x = vlad_x.view(x_size, -1)  
            vlad_x = F.normalize(vlad_x, p=2, dim=1)  
            neighborsFeat.append(vlad_x)
        node_features_list.append(neighborsFeat[NB])
        node_features_list.append(torch.concat(neighborsFeat[0:NB],0))
        neighborsFeat = []
        gvlad = self.applyGNN(node_features_list)
        gvlad = torch.add(gvlad,vlad_x)        
        gvlad = gvlad.view(-1,vlad_x.shape[1])
        N, D = gvlad.size()
        gvlad = gvlad.view(N, D, 1, 1)
        gvlad = self.pca_layer(gvlad).view(N, -1)
        gvlad = F.normalize(gvlad, p=2, dim=-1)  
        return gvlad   
class GraphVLADEmbedRegion(nn.Module):
    def __init__(self, base_model, net_vlad, tuple_size, fastscnn, NB):
        super(GraphVLADEmbedRegion, self).__init__()
        self.base_model = base_model
        self.net_vlad = net_vlad
        self.tuple_size = tuple_size
        self.fastscnn = fastscnn
        
        self.NB = NB
        self.applyGNN = applyGNN()
        self.mask = True

        self.SelectRegions = SelectRegions(self.NB, self.mask)
        
    def _init_params(self):
        self.base_model._init_params()
        self.net_vlad._init_params()

    def _compute_region_sim(self, feature_A, feature_B):
        # feature_A: B*C*H*W
        # feature_B: (B*(1+neg_num))*C*H*W

        def reshape(x):
            # re-arrange local features for aggregating quarter regions
            N, C, H, W = x.size()
            x = x.view(N, C, 2, int(H/2), 2, int(W/2))
            x = x.permute(0,1,2,4,3,5).contiguous()
            x = x.view(N, C, -1, int(H/2), int(W/2))
            return x

        feature_A = reshape(feature_A)
        feature_B = reshape(feature_B)

        # compute quarter-region features
        def aggregate_quarter(x):
            N, C, B, H, W = x.size()
            x = x.permute(0,2,1,3,4).contiguous()
            x = x.view(-1,C,H,W)
            vlad_x = self.net_vlad(x) # (N*B)*64*512
            _, cluster_num, feat_dim = vlad_x.size()
            vlad_x = vlad_x.view(N,B,cluster_num,feat_dim)
            return vlad_x

        vlad_A_quarter = aggregate_quarter(feature_A)
        vlad_B_quarter = aggregate_quarter(feature_B)

        # compute half-region features
        def quarter_to_half(vlad_x):
            return torch.stack((vlad_x[:,0]+vlad_x[:,1], vlad_x[:,2]+vlad_x[:,3], \
                                vlad_x[:,0]+vlad_x[:,2], vlad_x[:,1]+vlad_x[:,3]), dim=1).contiguous()

        vlad_A_half = quarter_to_half(vlad_A_quarter)
        vlad_B_half = quarter_to_half(vlad_B_quarter)

        # compute global-image features
        def quarter_to_global(vlad_x):
            return vlad_x.sum(1).unsqueeze(1).contiguous()

        vlad_A_global = quarter_to_global(vlad_A_quarter)
        vlad_B_global = quarter_to_global(vlad_B_quarter)

        def norm(vlad_x):
            N, B, C, _ = vlad_x.size()
            vlad_x = F.normalize(vlad_x, p=2, dim=3)  # intra-normalization
            vlad_x = vlad_x.view(N, B, -1)  # flatten
            vlad_x = F.normalize(vlad_x, p=2, dim=2)  # L2 normalize
            return vlad_x

        vlad_A = torch.cat((vlad_A_global, vlad_A_half, vlad_A_quarter), dim=1)
        vlad_B = torch.cat((vlad_B_global, vlad_B_half, vlad_B_quarter), dim=1)
        vlad_A = norm(vlad_A)
        vlad_B = norm(vlad_B)

        _, B, L = vlad_B.size()
        vlad_A = vlad_A.view(self.tuple_size,-1,B,L)
        vlad_B = vlad_B.view(self.tuple_size,-1,B,L)

        score = torch.bmm(vlad_A.expand_as(vlad_B).view(-1,B,L), vlad_B.view(-1,B,L).transpose(1,2))
        score = score.view(self.tuple_size,-1,B,B)

        return score, vlad_A, vlad_B

    def _forward_train(self, x):
        B, C, H, W = x.size()
        x = x.view(self.tuple_size, -1, C, H, W)

        anchors = x[:, 0].unsqueeze(1).contiguous().view(-1,C,H,W) # B*C*H*W
        pairs = x[:, 1:].view(-1,C,H,W) # (B*(1+neg_num))*C*H*W

        return self._compute_region_sim(anchors, pairs)

    def forward(self, x):
        if (not self.training):
            node_features_list = []
            neighborsFeat = []

            pool_x, x_size, x_nodes = self.SelectRegions(x, self.base_model, self.fastscnn)


            for i in range(self.NB+1):
                vlad_x = self.net_vlad(x_nodes[i])
                vlad_x = F.normalize(vlad_x, p=2, dim=2)  
                vlad_x = vlad_x.view(x_size, -1)  
                vlad_x = F.normalize(vlad_x, p=2, dim=1)  
                neighborsFeat.append(vlad_x)
                
            node_features_list.append(neighborsFeat[self.NB])
            node_features_list.append(torch.concat(neighborsFeat[0:self.NB],0))
            del neighborsFeat
            gvlad = self.applyGNN(node_features_list)
            gvlad = torch.add(gvlad,vlad_x)
            gvlad = gvlad.view(-1,vlad_x.shape[1])
            
            # Clear node_features_list to free up memory
            del node_features_list
        
            return pool_x, gvlad
        else:
            pool_x, x = self.base_model(x)
            return self._forward_train(x)
        
