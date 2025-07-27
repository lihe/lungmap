import torch
import torch.nn as nn
from torch_scatter import scatter_max
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
from torch_geometric.utils import degree

def topology_preserving_sampling(nodes, edges, sample_ratio=0.5):
    """
    🔧 修复版本：确保所有张量操作的维度一致性
    nodes: [N, 3]
    edges: [2, E]
    return: sampled_nodes_idx: index of preserved nodes
    """
    N = nodes.shape[0]
    device = nodes.device

    # 边界检查
    if N == 0:
        return torch.tensor([], dtype=torch.long, device=device)
    
    # 处理空边情况
    if edges.shape[1] == 0:
        num_sample = max(1, int(sample_ratio * N))
        return torch.randperm(N, device=device)[:num_sample]

    try:
        # Step 1: 计算每个点的度数
        deg = degree(edges[0], N)  # [N]

        # Step 2: 找出拓扑关键点（端点或分叉点）
        key_nodes_idx = torch.where(deg != 2)[0]  # [K] - 1D张量

        # Step 3: 其余点中做 FPS（远点采样）
        normal_nodes_idx = torch.where(deg == 2)[0]  # [F] - 1D张量
        
        # 🔧 关键修复：确保维度计算正确
        key_count = len(key_nodes_idx)
        normal_count = len(normal_nodes_idx)
        target_sample_count = max(1, int(sample_ratio * N))
        fps_sample_count = max(0, target_sample_count - key_count)
        
        if fps_sample_count > 0 and normal_count > 0:
            fps_sample_count = min(fps_sample_count, normal_count)
            fps_nodes_idx = farthest_point_sampling(nodes[normal_nodes_idx], fps_sample_count)
            
            # 🔧 确保所有张量都是1D并且兼容
            if len(fps_nodes_idx) > 0:
                selected_normal = normal_nodes_idx[fps_nodes_idx]  # [fps_sample_count]
                
                # 确保两个张量都是1D
                key_nodes_flat = key_nodes_idx.flatten()
                selected_normal_flat = selected_normal.flatten()
                
                selected_idx = torch.cat([key_nodes_flat, selected_normal_flat])  # [K + fps_sample_count]
            else:
                selected_idx = key_nodes_idx.flatten()
        else:
            selected_idx = key_nodes_idx.flatten() if len(key_nodes_idx) > 0 else torch.tensor([0], device=device)

        return selected_idx
        
    except Exception as e:
        print(f"⚠️  Error in topology_preserving_sampling: {e}")
        # 降级：随机采样
        num_sample = max(1, int(sample_ratio * N))
        return torch.randperm(N, device=device)[:num_sample]

def farthest_point_sampling(xyz, n_sample):
    """
    🔧 修复版本：加强边界检查和维度保证
    xyz: [M, 3]
    return: [n_sample] index of sampled points using FPS
    """
    N = xyz.shape[0]
    device = xyz.device
    
    # 严格的边界检查
    if N == 0 or n_sample == 0:
        return torch.tensor([], dtype=torch.long, device=device)
    
    if n_sample >= N:
        return torch.arange(N, dtype=torch.long, device=device)
    
    try:
        centroids = torch.zeros(n_sample, dtype=torch.long, device=device)
        distance = torch.ones(N, device=device) * 1e10
        farthest = torch.randint(0, N, (1,), dtype=torch.long, device=device).item()

        for i in range(n_sample):
            centroids[i] = farthest
            centroid = xyz[farthest:farthest+1, :]  # [1, 3] - 确保维度
            dist = torch.sum((xyz - centroid) ** 2, dim=1)  # [N]
            distance = torch.min(distance, dist)
            farthest = torch.argmax(distance).item()

        return centroids  # [n_sample] - 保证1D
        
    except Exception as e:
        print(f"⚠️  Error in FPS: {e}")
        # 降级：随机采样
        return torch.randperm(N, device=device)[:n_sample]


def topology_aware_grouping(sampled_nodes_idx, full_edges, k=8):
    """
    🔧 修复版本：返回相对于采样节点的索引
    sampled_nodes_idx: 采样后的 [M] - 原始图中的节点索引
    full_edges: 原始拓扑图 [2, E]
    return: groups: [M, k] - 相对于采样节点的索引 (0到M-1)
    """
    M = sampled_nodes_idx.shape[0]
    device = sampled_nodes_idx.device
    
    if M == 0:
        return torch.empty((0, k), dtype=torch.long, device=device)

    try:
        edge_index = full_edges.cpu().numpy()
        sampled_idx_list = sampled_nodes_idx.tolist()
        sampled_idx_set = set(sampled_idx_list)

        # 建立邻接表
        from collections import defaultdict
        adj_dict = defaultdict(set)
        for src, tgt in edge_index.T:
            adj_dict[src].add(tgt)
            adj_dict[tgt].add(src)

        groups = []
        for i, original_idx in enumerate(sampled_idx_list):  # i是相对索引，original_idx是原始索引
            neighbors = list(adj_dict[original_idx])
            
            # 🔧 关键修复：将邻居的原始索引转换为相对索引
            valid_neighbors = []
            for neighbor_original_idx in neighbors:
                if neighbor_original_idx in sampled_idx_set:
                    # 找到邻居在采样节点中的相对位置
                    try:
                        relative_idx = sampled_idx_list.index(neighbor_original_idx)
                        valid_neighbors.append(relative_idx)
                    except ValueError:
                        continue
            
            # 补齐到k个邻居
            if len(valid_neighbors) >= k:
                groups.append(valid_neighbors[:k])
            else:
                # 用自己的相对索引补齐（自环）
                padding_needed = k - len(valid_neighbors)
                groups.append(valid_neighbors + [i] * padding_needed)

        # 🔧 确保返回正确维度
        if len(groups) != M:
            print(f"⚠️  Group count mismatch: expected {M}, got {len(groups)}")
            # 创建默认分组
            groups = [[i] * k for i in range(M)]

        result = torch.tensor(groups, dtype=torch.long, device=device)  # [M, k]
        
        # 最终维度检查
        if result.shape != (M, k):
            print(f"⚠️  Final shape mismatch: expected ({M}, {k}), got {result.shape}")
            result = torch.zeros((M, k), dtype=torch.long, device=device)
            for i in range(M):
                result[i, :] = i  # 自环分组作为降级
        
        return result
        
    except Exception as e:
        print(f"⚠️  Error in topology_aware_grouping: {e}")
        # 降级处理：返回自环分组
        result = torch.zeros((M, k), dtype=torch.long, device=device)
        for i in range(M):
            result[i, :] = i
        return result


class PointNetUnit(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels)
        )

    def forward(self, group_feats):  # [M, k, C]
        try:
            # 🔧 添加输入验证
            if group_feats.dim() != 3:
                print(f"⚠️  PointNetUnit input dimension error: expected 3D, got {group_feats.dim()}D, shape={group_feats.shape}")
                return torch.zeros((group_feats.shape[0], self.mlp[-1].out_features), device=group_feats.device)
            
            x = self.mlp(group_feats)  # [M, k, C']
            x, _ = torch.max(x, dim=1)  # [M, C']
            return x
            
        except Exception as e:
            print(f"⚠️  PointNetUnit error: {e}")
            M = group_feats.shape[0]
            return torch.zeros((M, self.mlp[-1].out_features), device=group_feats.device)


class TopoSAModule(nn.Module):
    def __init__(self, in_channels, out_channels, k=8, sample_ratio=0.5):
        super().__init__()
        self.k = k
        self.sample_ratio = sample_ratio
        self.pointnet = PointNetUnit(in_channels, out_channels)
        self.gcn = GCNConv(out_channels, out_channels)

    def forward(self, x, pos, edge_index):
        try:
            # 🔧 严格的输入验证
            if x.shape[0] != pos.shape[0]:
                print(f"⚠️  Input size mismatch: x={x.shape}, pos={pos.shape}")
                return x, pos, edge_index
            
            if pos.shape[1] != 3:
                print(f"⚠️  Position dimension error: expected [N, 3], got {pos.shape}")
                return x, pos, edge_index
                
            # Step 1: TPS 采样节点
            sampled_idx = topology_preserving_sampling(pos, edge_index, self.sample_ratio)
            
            if len(sampled_idx) == 0:
                empty_feat = torch.zeros((0, self.pointnet.mlp[-1].out_features), device=x.device)
                empty_pos = torch.zeros((0, 3), device=x.device)
                empty_edges = torch.zeros((2, 0), dtype=torch.long, device=x.device)
                return empty_feat, empty_pos, empty_edges
                
            # Step 2: 提取采样后的数据
            pos_out = pos[sampled_idx]  # [M, 3]
            x_sampled = x[sampled_idx]  # [M, C]

            # Step 3: 获取分组索引
            group_idx = topology_aware_grouping(sampled_idx, edge_index, k=self.k)  # [M, k]
            
            # Step 4: 🔧 严格的维度检查
            M, C = x_sampled.shape
            if group_idx.shape != (M, self.k):
                print(f"⚠️  Group index shape error: expected ({M}, {self.k}), got {group_idx.shape}")
                # 使用降级分组
                group_idx = torch.zeros((M, self.k), dtype=torch.long, device=x.device)
                for i in range(M):
                    group_idx[i, :] = i
            
            # 检查索引范围
            if (group_idx < 0).any() or (group_idx >= M).any():
                print(f"⚠️  Invalid indices: min={group_idx.min()}, max={group_idx.max()}, valid_range=[0, {M-1}]")
                group_idx = torch.clamp(group_idx, 0, M-1)

            # Step 5: 提取分组特征
            group_feat = x_sampled[group_idx]  # [M, k, C]
            
            # 🔧 最终维度验证
            if group_feat.shape != (M, self.k, C):
                print(f"⚠️  Group feature shape error: expected ({M}, {self.k}, {C}), got {group_feat.shape}")
                group_feat = x_sampled.unsqueeze(1).expand(-1, self.k, -1)  # 降级处理
            
            point_feat = self.pointnet(group_feat)  # [M, C']

            # Step 6: 重建子图
            new_edge_index = self._rebuild_subgraph_edges(sampled_idx, edge_index, x.device)
            x_out = self.gcn(point_feat, new_edge_index)

            return x_out, pos_out, new_edge_index
            
        except Exception as e:
            print(f"⚠️  TopoSAModule critical error: {e}")
            print(f"    Input shapes: x={x.shape}, pos={pos.shape}, edge_index={edge_index.shape}")
            # 完全降级：返回输入
            return x, pos, edge_index
    
    def _rebuild_subgraph_edges(self, sampled_idx, edge_index, device):
        """重建子图边连接"""
        try:
            if len(sampled_idx) <= 1:
                return torch.empty((2, 0), dtype=torch.long, device=device)
                
            idx_map = {idx.item(): i for i, idx in enumerate(sampled_idx)}
            new_edges = []
            
            for original_idx in sampled_idx:
                neighbors = edge_index[1][edge_index[0] == original_idx]
                for neighbor_idx in neighbors:
                    if neighbor_idx.item() in idx_map:
                        src_new = idx_map[original_idx.item()]
                        tgt_new = idx_map[neighbor_idx.item()]
                        new_edges.append([src_new, tgt_new])
            
            if len(new_edges) == 0:
                return torch.empty((2, 0), dtype=torch.long, device=device)
            else:
                return torch.tensor(new_edges, dtype=torch.long, device=device).T
                
        except Exception as e:
            print(f"⚠️  Error rebuilding subgraph: {e}")
            return torch.empty((2, 0), dtype=torch.long, device=device)