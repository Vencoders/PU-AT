import math
import sys
from pointnet2_ops._ext import gather_points
sys.path.append("../")
import torch
import torch.nn as nn
from torch.nn import Conv1d, Conv2d
from knn_cuda import KNN
from pointnet2_ops.pointnet2_utils import grouping_operation
import torch.nn.functional as F
import pointnet2_ops.pointnet2_utils as pointnet2




class get_edge_feature(nn.Module):
    """construct edge feature for each point
    Args:
        tensor: input a point cloud tensor,batch_size,num_dims,num_points
        k: int
    Returns:
        edge features: (batch_size,num_dims,num_points,k)
    """
    def __init__(self, k=16):
        super(get_edge_feature, self).__init__()
        self.KNN = KNN(k=k + 1, transpose_mode=False)
        self.k = k

    def forward(self, point_cloud):
        dist, idx = self.KNN(point_cloud, point_cloud)
        """
        idx is batch_size,k,n_points
        point_cloud is batch_size,n_dims,n_points
        point_cloud_neightbors is batch_size,n_dims,k,n_points
        """
        idx = idx[:, 1:, :]
        point_cloud_neighbors = grouping_operation(point_cloud, idx.contiguous().int())
        point_cloud_central = point_cloud.unsqueeze(2).repeat(1, 1, self.k, 1)
        edge_feature = torch.cat(
            [point_cloud_central, point_cloud_neighbors - point_cloud_central], dim=1
        )
        return edge_feature, idx

class denseconv(nn.Module):
    def __init__(self, growth_rate=64, k=16, in_channels=6):
        super(denseconv, self).__init__()
        self.k = k
        self.edge_feature_model = get_edge_feature(k=k)
        """
        input to conv1 is batch_size,2xn_dims,k,n_points
        """
        self.conv1 = nn.Sequential(
            Conv2d(
                in_channels=2 * in_channels,
                out_channels=growth_rate,
                kernel_size=[1, 1],
            ),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            Conv2d(
                in_channels=growth_rate + in_channels,
                out_channels=growth_rate,
                kernel_size=[1, 1],
            ),
            nn.ReLU(),
        )
        self.conv3 = nn.Sequential(
            Conv2d(
                in_channels=2 * growth_rate + in_channels,
                out_channels=growth_rate,
                kernel_size=[1, 1],
            ),
        )

    def forward(self, input):
        """
        y should be batch_size,in_channel,k,n_points
        """
        y, idx = self.edge_feature_model(input)  # B c k n
        inter_result = torch.cat(
            [self.conv1(y), input.unsqueeze(2).repeat([1, 1, self.k, 1])], dim=1
        )
        inter_result = torch.cat([self.conv2(inter_result), inter_result], dim=1)
        inter_result = torch.cat([self.conv3(inter_result), inter_result], dim=1)
        final_result = torch.max(inter_result, dim=2)[0]  # pool the k channel
        return final_result, idx


class feature_extraction(nn.Module):
    def __init__(self, growth_rate=24, dense_n=3, k=16):
        super(feature_extraction, self).__init__()
        self.growth_rate = growth_rate
        self.dense_n = dense_n
        self.k = k
        self.input_channel = 3
        comp = self.growth_rate * 2
        """
        make sure to permute the input, the feature dimension is in the second one.
        input of conv1 is batch_size,num_dims,num_points
        """
        self.conv1 = nn.Sequential(
            Conv1d(
                in_channels=self.input_channel,
                out_channels=24,
                kernel_size=1,
                padding=0,
            ),
            nn.ReLU(),
        )
        self.denseconv1 = denseconv(
            in_channels=24, growth_rate=self.growth_rate, k=self.k
        )
        self.conv2 = nn.Sequential(
            Conv1d(in_channels=120, out_channels=comp, kernel_size=1), nn.ReLU()
        )
        self.denseconv2 = denseconv(
            in_channels=comp, growth_rate=self.growth_rate, k=self.k
        )
        self.conv3 = nn.Sequential(
            Conv1d(in_channels=240, out_channels=comp, kernel_size=1), nn.ReLU()
        )
        self.denseconv3 = denseconv(
            in_channels=comp, growth_rate=self.growth_rate, k=self.k
        )
        self.conv4 = nn.Sequential(
            Conv1d(in_channels=360, out_channels=comp, kernel_size=1), nn.ReLU()
        )
        self.denseconv4 = denseconv(
            in_channels=comp, growth_rate=self.growth_rate, k=self.k
        )

    def forward(self, input):
        l0_features = self.conv1(input)  # b,24,n
        l1_features, l1_index = self.denseconv1(l0_features)  # b,24*2+24*3=120,n
        l1_features = torch.cat([l1_features, l0_features], dim=1)  # b,120,n
        l2_features = self.conv2(l1_features)  # b,48,n
        l2_features, l2_index = self.denseconv2(l2_features)
        l2_features = torch.cat([l2_features, l1_features], dim=1)  # b,240,n
        l3_features = self.conv3(l2_features)  # b,48,n
        l3_features, l3_index = self.denseconv3(l3_features)
        l3_features = torch.cat([l3_features, l2_features], dim=1)  # b,360,n
        l4_features = self.conv4(l3_features)  # b,48,n
        l4_features, l4_index = self.denseconv4(l4_features)
        l4_features = torch.cat([l4_features, l3_features], dim=1)  # b,480,n
        return l4_features

    class up_block(nn.Module):
        def __init__(self, up_ratio=4, in_channels=130, device=None):
            super(up_block, self).__init__()
            self.up_ratio = up_ratio
            self.conv1 = nn.Sequential(
                Conv1d(in_channels=in_channels, out_channels=128, kernel_size=1), nn.ReLU()
            )
            self.conv2 = nn.Sequential(
                Conv1d(in_channels=128, out_channels=128, kernel_size=1), nn.ReLU()
            )
            self.grid = self.gen_grid(up_ratio).clone().detach()
            self.attention_unit = attention_unit(in_channels=in_channels)

        def forward(self, inputs):
            net = inputs  
            grid = self.grid.clone().to(net.device)
            grid = grid.unsqueeze(0).repeat(net.shape[0], 1, net.shape[2])  
            grid = grid.view([net.shape[0], -1, 2])  
            net = net.permute(0, 2, 1).contiguous()  
            net = net.repeat(1, self.up_ratio, 1)  
            net = torch.cat([net, grid], dim=2)  
            net = net.permute(0, 2, 1).contiguous()  
            net = self.attention_unit(net)
            net = self.conv1(net)
            net = self.conv2(net)
            return net
    def gen_grid(self, up_ratio):
        import math
        sqrted = int(math.sqrt(up_ratio)) + 1
        for i in range(1, sqrted + 1).__reversed__():
            if (up_ratio % i) == 0:
                num_x = i
                num_y = up_ratio // i
                break
        grid_x = torch.linspace(-0.2, 0.2, num_x)
        grid_y = torch.linspace(-0.2, 0.2, num_y)
        x, y = torch.meshgrid([grid_x, grid_y])
        grid = torch.stack([x, y], dim=-1)  
        grid = grid.view([-1, 2])  
        return grid


class refinerfea(nn.Module):
    def __init__(self, K1=16, K2=8, transform_dim=64,in_channel=128):
        super(refinerfea, self).__init__()
        self.K1 = K1 + 1
        self.KNN = KNN(self.K1)
        self.in_channel = in_channel
        self.transform_dim = transform_dim
        self.gamma_dim = self.transform_dim
        #self.conv1 = nn.Conv2d(3, 32 ,1)
        self.atten = attention_unit(64)
        self.conv2 = nn.Sequential(
            nn.Conv1d(64, 64, 1),
            nn.ReLU(),
            nn.Conv1d(64, 3, 1))

    def forward(self, feature, xyz):  
        # b n c
        dis_, idx = self.KNN(xyz, xyz)  
        idx = idx[:, 1:, :]            
        dis = dis_[:, 1:, :]            
        dis_new = 1 / dis
        dis_all = torch.sum(dis_new,dim=1) 
        dis_all = dis_all.unsqueeze(1).repeat(1,  self.K1-1, 1)
        weight = dis_new / (dis_all + 0.0000001)  
        weight = weight.unsqueeze(1) 
        group_xyz = grouping_operation(feature, idx.contiguous().int())  
        res = weight * group_xyz  
        fea2 = torch.max(res, dim=2)[0]  
        fea2 = self.atten(fea2)
        fea3 = self.conv2(fea2)
        return fea3


class MLP_Res(nn.Module):
    def __init__(self, in_dim=128, hidden_dim=None, out_dim=128):
        super(MLP_Res, self).__init__()
        if hidden_dim is None:
            hidden_dim = in_dim
        self.conv_1 = nn.Conv1d(in_dim, hidden_dim, 1)
        self.conv_2 = nn.Conv1d(hidden_dim, out_dim, 1)
        self.conv_shortcut = nn.Conv1d(in_dim, out_dim, 1)

    def forward(self, x):
        """
        Args:
            x: (B, out_dim, n)
        """
        shortcut = self.conv_shortcut(x)
        out = self.conv_2(torch.relu(self.conv_1(x))) + shortcut
        return out


class localTransformer(nn.Module):
    def __init__(self, in_channel=480, transform_dim=64,k=16):
        super(localTransformer, self).__init__()
        self.transform_dim = transform_dim
        self.k = k
        self.KNN = KNN(k=k + 1, transpose_mode=False)
        self.bottleneck = MLP_Res(in_dim=480, hidden_dim=64, out_dim=64) 
        self.w_qs = nn.Conv1d(self.transform_dim, self.transform_dim, 1)
        self.w_ks = nn.Conv1d(self.transform_dim, self.transform_dim, 1)
        self.w_vs = nn.Conv1d(self.transform_dim, self.transform_dim, 1)
        self.fc_delta = nn.Sequential(
            nn.Conv2d(3, 64, [1, 1]),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, self.transform_dim, [1, 1])
        )
        self.fc_gamma = nn.Sequential(
            nn.Conv2d(self.transform_dim, 4 * self.transform_dim, [1, 1]),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(4 * self.transform_dim, self.transform_dim, [1, 1])
        )
        self.atten_mlp = nn.ConvTranspose2d(64, 64, (4,1), (4,1))
        self.upsample1 = nn.Upsample(scale_factor=(4,1))
        self.upsample2 = nn.Upsample(scale_factor=4)
        self.mlp_1 = MLP_Res(in_dim=32 + 64, hidden_dim=128, out_dim=128)
        self.mlp_2 = MLP_Res(in_dim=128, hidden_dim=64, out_dim=128)
        self.mlp_3 = MLP_Res(in_dim=128 + 32, hidden_dim=128, out_dim=128)
        self.mlp_4 = nn.Sequential(
            nn.Conv1d(64, 64, 1),
            nn.ReLU(),
            nn.Conv1d(64, 3, 1))
            
    def forward(self, feature, xyz):
        dist, idx = self.KNN(xyz, xyz)
        idx = idx[:, 1:, :]
        x = self.bottleneck(feature)  
        q, k, v = self.w_qs(x), self.w_ks(x), self.w_vs(x)  
        group_xyz = grouping_operation(xyz, idx.contiguous().int())  
        rel_xyz = xyz[:, :, None, :] - group_xyz  
        pos_enc = self.fc_delta(rel_xyz)  
        k_local = grouping_operation(k, idx.contiguous().int())
        v_local = grouping_operation(v, idx.contiguous().int())  
        qk_rel = q[:, :, None, :] - k_local
        attn = self.fc_gamma(qk_rel + pos_enc) 
        attn1 = attn.permute(0, 1,3,2).contiguous() 
        attn = self.atten_mlp(attn1) 
        attn = F.softmax(attn, dim=1)  
        value_rel = (v_local + pos_enc).permute(0, 1,3,2).contiguous() 
        value = self.upsample1(value_rel) 
        res = torch.einsum("bmnf,bmnf->bmn", attn, value) 
        identity = self.upsample2(x)  
        res = res + identity
        completion = self.mlp_4(res)
        return completion, res


class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.args = args
        self.num_point = args.num_point 
        self.KNN = KNN(k=17, transpose_mode=False)     
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=1),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=3, kernel_size=1),
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=64, kernel_size=1),
            nn.ReLU()
        )
        self.conv4 = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=3, kernel_size=1),
        )
        self.mpu_extraction = feature_extraction(growth_rate=24, dense_n=3, k=16) 
        self.seed_generator = localTransformer(in_channel=32, transform_dim=64,k=16)
        self.refiner = refinerfea(K1=args.K1, K2=args.K2, transform_dim=64, in_channel=64)
        self.conv5 = nn.Sequential(
            nn.Conv1d(in_channels=3, out_channels=32, kernel_size=1),
            nn.ReLU()
        )
        self.conv2d1 = nn.Conv2d(6, 32, 1)
        self.gcn = edge_conv(3, 32)
    def forward(self, inputs):
        xyz = inputs.permute(0, 2, 1).contiguous()
        fea = self.mpu_extraction(inputs) 
        coarse_coord, H = self.seed_generator(fea, inputs)
        refine = self.refiner(H, coarse_coord) 
        refine_coord = refine + coarse_coord
        return coarse_coord, refine_coord

        
class node_shuffle(nn.Module):
    def __init__(self, scale=2, in_channels=128, out_channels=128):
        super(node_shuffle, self).__init__()
        self.scale = scale
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels = in_channels, out_channels=out_channels, kernel_size=1), nn.ReLU()
        )  # 480 256
        self.edge_conv = edge_conv(out_channels, out_channels * scale)

    def forward(self, inputs):
        B, C, N = inputs.shape[0], inputs.shape[1], inputs.shape[2]
        net = inputs  # b,480,n
        net = self.conv(net)  # 128
        net = self.edge_conv(net)  # b out_channel, 1 ,n
        net = net.squeeze(-2).contiguous()
        net = net.reshape([B, -1, self.scale * N]).contiguous()
        return net

class edge_conv(nn.Module):
    def __init__(self, in_channels, out_channels, k=16):
        super(edge_conv, self).__init__()
        self.k = k
        self.KNN = KNN(self.k + 1)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1)
        self.conv2 = nn.Conv2d(in_channels, out_channels, 1)
        self.relu = nn.ReLU()

    def forward(self, inputs):  # b c n
        _, idx = self.KNN(inputs, inputs)
        idx = idx[:, 1:, :]
        pc_neighbors = grouping_operation(inputs, idx.contiguous().int())  # b c k n
        inputs = inputs.unsqueeze(-2).contiguous()
        pc_central = inputs.repeat([1, 1, self.k, 1]).contiguous()
        message = self.conv1(pc_neighbors - pc_central)
        x_center = self.conv2(inputs)
        edge_features = x_center + message
        edge_features = self.relu(edge_features)
        y = torch.max(edge_features, -2, keepdims=True)[0]
        return y

class attention_unit(nn.Module):
    def __init__(self, in_channels=64):
        super(attention_unit, self).__init__()
        self.convF = nn.Sequential(
            Conv1d(
                in_channels=in_channels, out_channels=in_channels // 4, kernel_size=1
            ),
            nn.ReLU(),
        )
        self.convG = nn.Sequential(
            Conv1d(
                in_channels=in_channels, out_channels=in_channels // 4, kernel_size=1
            ),
            nn.ReLU(),
        )
        self.convH = nn.Sequential(
            Conv1d(in_channels=in_channels, out_channels=in_channels, kernel_size=1),
            nn.ReLU(),
        )
        self.gamma = nn.Parameter(
            torch.zeros([1]).clone().detach().requires_grad_(True)
        )

    def forward(self, inputs):
        f = self.convF(inputs)
        g = self.convG(inputs)  # b,32,n
        h = self.convH(inputs)
        s = torch.matmul(g.permute(0, 2, 1).contiguous(), f)  # b,n,n
        beta = F.softmax(s, dim=2)  # b,n,n
        o = torch.matmul(h, beta)  # b,130,n
        x = self.gamma * o + inputs
        return x

class up_block(nn.Module):
    def __init__(self, up_ratio=4, in_channels=130, device=None):
        super(up_block, self).__init__()
        self.up_ratio = up_ratio
        self.conv1 = nn.Sequential(
            Conv1d(in_channels=in_channels, out_channels=256, kernel_size=1), nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            Conv1d(in_channels=256, out_channels=128, kernel_size=1), nn.ReLU()
        )
        self.grid = self.gen_grid(up_ratio).clone().detach()
        self.attention_unit = attention_unit(in_channels=in_channels)

    def forward(self, inputs):
        net = inputs  # b,128,n
        grid = self.grid.clone().to(net.device)
        grid = grid.unsqueeze(0).repeat(net.shape[0], 1, net.shape[2])  # b,4,2*n
        grid = grid.view([net.shape[0], -1, 2])  # b,4*n,2
        net = net.permute(0, 2, 1).contiguous()  # b,n,128
        net = net.repeat(1, self.up_ratio, 1)  # b,4n,128
        net = torch.cat([net, grid], dim=2)  # b,n*4,130
        net = net.permute(0, 2, 1).contiguous()  # b,130,n*4
        net = self.attention_unit(net)
        net = self.conv1(net)
        net = self.conv2(net)
        return net

    def gen_grid(self, up_ratio):
        import math
        sqrted = int(math.sqrt(up_ratio)) + 1
        for i in range(1, sqrted + 1).__reversed__():
            if (up_ratio % i) == 0:
                num_x = i
                num_y = up_ratio // i
                break
        grid_x = torch.linspace(-0.2, 0.2, num_x)
        grid_y = torch.linspace(-0.2, 0.2, num_y)
        x, y = torch.meshgrid([grid_x, grid_y])
        grid = torch.stack([x, y], dim=-1)  # 2,2,2
        grid = grid.view([-1, 2])  # 4,2
        return grid

if __name__ == "__main__":
    import sys
    sys.path.append("../")
    from common.configs import args
    from time import time
    from thop import profile
    f = Model(args).cuda()
    times = []
    for i in range(100):
        a = torch.randn([2, 3, 256])
        a = a.float().cuda()
        start = time()
        result = f(a)
        end = time()
        times.append((end - start) / 2)
        print((end - start))
    flops, params = profile(f, inputs=(a,))
    print(flops / 1024 / 1024 / 1024 / 2, params / 1024 / 1024 * 4)
    para_num = sum([p.numel() for p in f.parameters()])
    print("=== The number of parameters in model: {:.4f} K === ".format(float(para_num / 1e3)))

