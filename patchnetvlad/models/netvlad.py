import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import copy
import code
#code.interact(local=locals())

# graphsage
import torch.nn.init as init

# Semantic Segmentation
from torchvision import transforms
# import espnet as net
from .espnet import *
from torchvision.ops import masks_to_boxes


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
        subfeat_size = int(hidden[0].shape[1]/self.input_dim)
        gcndim = int(self.input_dim) 
        
        # print('subfeat_size ', subfeat_size)
        # print('  l  ', ' hop  ', '  src_node_features  ', '  neighbor_node_features  ', '  h  ', '    ')

        for l in range(self.num_layers):
            next_hidden = []
            gcn = self.gcn[l]
            for hop in range(self.num_layers - l):
                src_node_features = hidden[hop]
                src_node_num = len(src_node_features)
                # print('neighbor_node_features ', hidden[hop + 1].shape  ,' / ',  src_node_num, self.num_neighbors_list[hop], '-1')
                neighbor_node_features = hidden[hop + 1] \
                    .view((src_node_num, self.num_neighbors_list[hop], -1))
                # print(l,' ', hop  ,'  ',  src_node_features.shape  ,'  ' , neighbor_node_features.shape)
                
                # splitting the i/p
                #h = gcn(src_node_features, neighbor_node_features)
                for j in range(subfeat_size): 
                    h_x = gcn(src_node_features[:,j*gcndim:j*gcndim+gcndim], neighbor_node_features[:,:,j*gcndim:j*gcndim+gcndim])
                    # neighborsFeat = []
                    if (j==0):
                        h = h_x;
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
        
        # Semantic Segmentation
        self.classes = 20
        self.p = 2
        self.q = 8
        self.encoderFile = "/media/leo/2C737A9872F69ECF/datasets/pretrained/encoder/espnet_p_2_q_8.pth"
        self.Espnet = ESPNet(classes=self.classes, p=self.p, q = self.q, encoderFile=self.encoderFile)  # Net.Mobile_SegNetDilatedIA_C_stage1(20)
        # requires_grad=False
        
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
        
        # createboxes
        # print("debuggind started")
        with torch.no_grad():
            b_out = self.Espnet(x)
        # b_out = self.Espnet(x)
        mask = b_out.max(1)[1]   #torch.Size([36, 480, 640])
        
        for jj in range(len(mask)):  #batch processing
            
            single_label_mask = mask[jj]
            
            # obj_ids = torch.unique(single_label_mask)
            obj_ids, obj_i = single_label_mask.unique(return_counts=True)
            obj_ids = obj_ids[1:] 
            obj_i = obj_i[1:]
            #torch.Size([19])
            
            masks = single_label_mask == obj_ids[:, None, None]
            boxes_t = masks_to_boxes(masks.to(torch.float32))

            
            rr_boxes = torch.argsort(torch.argsort(obj_i,descending=True)) # (decending order)

    
            
            boxes = boxes_t/16
        
  
        _, _, H, W = x.shape
        patch_mask = torch.zeros((H, W)).cuda()
        
        # VGG 
        pool_x, x = self.base_model(x)   
        
        N, C, H, W = x.shape

        
        # img_orig = to_tensor(Image.open(image_list[jj]).convert('RGB'))
        # _, H, W = img_orig.shape

        bb_x = [[int(W/4), int(H/4), int(3*W/4),int(3*H/4)],
                [0, 0, int(W/3),H], 
                [0, 0, W,int(H/3)], 
                [int(2*W/3), 0, W,H], 
                [0, int(2*H/3), W,H]]

 
        NB = 5
        
        graph_nodes = torch.zeros(N,NB,C,H,W).cuda()
        rsizet = transforms.Resize((H,W)) #H W

        
        for Nx in range(N):    
            # img_stk = x[Nx].unsqueeze(0)
            img_nodes = []
            # print(Nx)
            for idx in range(len(boxes)):
                for b_idx in range(len(rr_boxes)):


                    if idx == rr_boxes[b_idx] and obj_i[b_idx] > 10000 and len(img_nodes) < NB-1:

                        patch_mask = patch_mask*0

                        # label obj_ids[rr_boxes[b_idx]]
                        patch_mask[single_label_mask == obj_ids[b_idx]] = 1
                        # box boxes[rr_boxes[b_idx]]

                        # patch_mask = patch_mask.unsqueeze(0)
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
            # code.interact(local=locals())
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
        
        ## Graphsage
        gvlad = self.graph(node_features_list)

        gvlad = torch.add(gvlad,vlad_x)

        
        return pool_x, gvlad.view(-1,32768)

class EmbedNetPCA(nn.Module):
    def __init__(self, base_model, net_vlad, dim=4096):
        super(EmbedNetPCA, self).__init__()
        self.base_model = base_model
        self.net_vlad = net_vlad
        self.pca_layer = nn.Conv2d(net_vlad.num_clusters*net_vlad.dim, dim, 1, stride=1, padding=0)

    def _init_params(self):
        self.base_model._init_params()
        self.net_vlad._init_params()

    def forward(self, x):
        _, x = self.base_model(x)
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
