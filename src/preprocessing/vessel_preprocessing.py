import os
import numpy as np
import SimpleITK as sitk
from scipy import ndimage
from skimage.morphology import skeletonize_3d
from sklearn.neighbors import NearestNeighbors
from typing import Dict, List, Tuple, Optional
import networkx as nx
from scipy.spatial.distance import pdist, squareform

class VesselPreprocessor:
    """血管预处理器：从分割标签到图构建的完整pipeline"""
    
    def __init__(self, 
                 ct_dir: str,
                 label_dir: str,
                 output_dir: str,
                 spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
                 cube_size: int = 32):
        self.ct_dir = ct_dir
        self.label_dir = label_dir
        self.output_dir = output_dir
        self.spacing = spacing
        self.cube_size = cube_size
        
        # 血管层次结构 - 对应CPR-TaG-Net的18个类别
        self.vessel_hierarchy = {
            'main': ['MPA'],  # 主肺动脉 - 类别0
            'primary': ['LPA', 'RPA'],  # 左右肺动脉 - 类别1,2
            'secondary_left': ['Linternal', 'Lupper', 'Lmedium', 'Ldown'],  # 左侧二级 - 类别3-6
            'secondary_right': ['Rinternal', 'Rupper', 'Rmedium', 'RDown'],  # 右侧二级 - 类别7-10
            'tertiary_left': ['L1+2', 'L1+3'],  # 左侧三级 - 类别11-12
            'tertiary_right': ['R1+2', 'R1+3'],  # 右侧三级 - 类别13-14
            'background': ['background'],  # 背景 - 类别15
            'uncertain': ['uncertain'],  # 不确定 - 类别16
            'junction': ['junction']  # 连接点 - 类别17
        }
        
        # 创建标签到类别的映射
        self.label_to_class = {}
        class_id = 0
        for level, vessels in self.vessel_hierarchy.items():
            for vessel in vessels:
                self.label_to_class[vessel] = class_id
                class_id += 1
        
        os.makedirs(output_dir, exist_ok=True)
    
    def process_case(self, case_id: str) -> Dict:
        """处理单个病例"""
        print(f"Processing case: {case_id}")
        
        # 加载数据
        ct_path = os.path.join(self.ct_dir, f"{case_id}.nii")
        
        # 查找匹配的标签文件（文件名以case_id开头）
        label_path = None
        for filename in os.listdir(self.label_dir):
            if filename.startswith(case_id) and filename.endswith('.seg.nrrd'):
                label_path = os.path.join(self.label_dir, filename)
                break
        
        if not os.path.exists(ct_path) or label_path is None:
            print(f"Missing files for case {case_id}")
            print(f"  CT path: {ct_path} (exists: {os.path.exists(ct_path)})")
            print(f"  Label path: {label_path}")
            return None
        
        ct_image = sitk.ReadImage(ct_path)
        label_image = sitk.ReadImage(label_path)
        
        ct_array = sitk.GetArrayFromImage(ct_image)
        label_array = sitk.GetArrayFromImage(label_image)
        
        # 🔧 处理4D数据：去除channel维度，当做灰度图处理
        print(f"Original shapes - CT: {ct_array.shape}, Label: {label_array.shape}")
        
        # 处理CT数组的维度 - 去除channel维度
        if ct_array.ndim == 4:
            print(f"CT array is 4D, removing channel dimension...")
            # 无论有几个通道，都只取第一个通道作为灰度图
            ct_array = ct_array[..., 0]
            print(f"Converted to 3D grayscale: {ct_array.shape}")
        
        # 处理标签数组的维度 - 去除channel维度
        if label_array.ndim == 4:
            print(f"Label array is 4D, removing channel dimension...")
            # 对于标签，取第一个通道
            label_array = label_array[..., 0]
            print(f"Converted to 3D: {label_array.shape}")
        
        # 确保都是3D数组
        if ct_array.ndim != 3 or label_array.ndim != 3:
            print(f"Error: Expected 3D arrays, got CT: {ct_array.ndim}D, Label: {label_array.ndim}D")
            return None
        
        # 确保标签数组是整数类型
        if not np.issubdtype(label_array.dtype, np.integer):
            label_array = label_array.astype(np.int32)
        
        print(f"Final 3D shapes - CT: {ct_array.shape}, Label: {label_array.shape}")
        
        # 存储ct_array以供后续使用
        self.ct_array = ct_array
        
        # 获取标签映射
        label_mapping = self._get_label_mapping_from_file(label_path)
        
        # 1. 提取中心线
        centerlines = self._extract_centerlines(label_array, label_mapping)
        
        if not centerlines:
            print(f"No valid centerlines found for case {case_id}")
            return None
        
        # 2. 构建血管图
        vessel_graph = self._build_vessel_graph(centerlines, ct_array, ct_image)
        
        # 3. 采样图像块
        image_cubes = self._sample_image_cubes(vessel_graph['nodes'], ct_array)
        
        # 4. 准备训练数据
        training_data = self._prepare_training_data(vessel_graph, image_cubes, case_id)
        
        return training_data
    
    def _get_label_mapping_from_file(self, label_path: str) -> Dict[str, int]:
        """从.seg.nrrd文件获取标签映射"""
        try:
            reader = sitk.ImageFileReader()
            reader.SetFileName(label_path)
            reader.LoadPrivateTagsOn()
            reader.ReadImageInformation()
            
            label_mapping = {}
            for i in range(200):
                name_key = f'Segment{i}_Name'
                value_key = f'Segment{i}_LabelValue'
                
                if reader.HasMetaDataKey(name_key) and reader.HasMetaDataKey(value_key):
                    name = reader.GetMetaData(name_key)
                    value = int(reader.GetMetaData(value_key))
                    label_mapping[name] = value
            
            return label_mapping
        except Exception as e:
            print(f"Error reading label mapping: {e}")
            return {}
    
    def _extract_centerlines(self, label_array: np.ndarray, label_mapping: Dict[str, int]) -> Dict:
        """提取血管中心线"""
        centerlines = {}
        
        print(f"Processing {len(label_mapping)} labels...")
        
        for vessel_name, label_value in label_mapping.items():
            if vessel_name in self.label_to_class:
                try:
                    print(f"  Processing vessel: {vessel_name}")
                    # 提取该标签的二值掩码
                    vessel_mask = (label_array == label_value).astype(np.uint8)
                    
                    if np.sum(vessel_mask) < 10:  # 太小的区域跳过
                        print(f"    Skipping: too small ({np.sum(vessel_mask)} voxels)")
                        continue
                    
                    # vessel_mask应该已经是3D的（在process_case中已经处理了4D->3D转换）
                    if vessel_mask.ndim != 3:
                        print(f"    Error: mask is not 3D (shape: {vessel_mask.shape})")
                        continue
                    
                    # 形态学清理
                    cleaned_mask = self._clean_vessel_mask(vessel_mask)
                    
                    if np.sum(cleaned_mask) < 5:
                        print(f"    Skipping: too small after cleaning")
                        continue
                    
                    # 提取3D骨架
                    skeleton = skeletonize_3d(cleaned_mask.astype(bool))
                    
                    # 获取骨架点
                    skeleton_coords = np.array(np.where(skeleton)).T
                    
                    if len(skeleton_coords) < 5:  # 骨架点太少跳过
                        print(f"    Skipping: too few skeleton points ({len(skeleton_coords)})")
                        continue
                    
                    # 排序骨架点（沿血管方向）
                    ordered_coords = self._order_skeleton_points(skeleton_coords)
                    
                    # 计算血管半径
                    radii = self._estimate_vessel_radii(ordered_coords, cleaned_mask)
                    
                    # 计算几何特征（简化版本）
                    features = self._compute_geometric_features(ordered_coords, radii)
                    
                    centerlines[vessel_name] = {
                        'coords': ordered_coords,
                        'radii': radii,
                        'features': features,
                        'label_value': label_value,
                        'class_id': self.label_to_class[vessel_name]
                    }
                    
                    print(f"    Success: {len(ordered_coords)} points")
                    
                except Exception as e:
                    print(f"    Error processing {vessel_name}: {e}")
                    continue
        
        print(f"Successfully extracted {len(centerlines)} centerlines")
        return centerlines
    
    def _clean_vessel_mask(self, mask: np.ndarray) -> np.ndarray:
        """清理血管掩码"""
        # 移除小的连通分量
        labeled_mask, num_labels = ndimage.label(mask)
        
        if num_labels == 0:
            return mask
        
        # 保留最大的连通分量
        label_sizes = np.bincount(labeled_mask.flat)
        largest_label = np.argmax(label_sizes[1:]) + 1
        
        cleaned_mask = (labeled_mask == largest_label).astype(np.uint8)
        
        # 形态学开运算去噪
        from scipy.ndimage import binary_opening
        cleaned_mask = binary_opening(cleaned_mask, structure=np.ones((3, 3, 3)))
        
        return cleaned_mask.astype(np.uint8)
    
    def _order_skeleton_points(self, coords: np.ndarray) -> np.ndarray:
        """沿血管方向排序骨架点"""
        if len(coords) <= 2:
            return coords
        
        # 使用最小生成树方法排序
        distances = squareform(pdist(coords))
        
        # 构建图
        G = nx.Graph()
        n = len(coords)
        
        # 只连接距离较近的点
        threshold = np.percentile(distances[distances > 0], 10)  # 取较小的距离作为阈值
        
        for i in range(n):
            for j in range(i + 1, n):
                if distances[i, j] <= threshold:
                    G.add_edge(i, j, weight=distances[i, j])
        
        if len(G.edges) == 0:
            return coords
        
        # 找到最小生成树
        mst = nx.minimum_spanning_tree(G)
        
        # 找到端点（度为1的节点）
        end_points = [node for node in mst.nodes() if mst.degree(node) == 1]
        
        if len(end_points) < 2:
            return coords
        
        # 从一个端点开始遍历
        start_node = end_points[0]
        path = list(nx.dfs_preorder_nodes(mst, source=start_node))
        
        return coords[path]
    
    def _estimate_vessel_radii(self, coords: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """估计血管半径"""
        radii = []
        
        for coord in coords:
            z, y, x = coord.astype(int)
            
            # 在该点周围估计半径
            radius = self._estimate_radius_at_point(mask, z, y, x)
            radii.append(radius)
        
        return np.array(radii)
    
    def _estimate_radius_at_point(self, mask: np.ndarray, z: int, y: int, x: int, max_radius: int = 10) -> float:
        """在指定点估计血管半径"""
        if not (0 <= z < mask.shape[0] and 0 <= y < mask.shape[1] and 0 <= x < mask.shape[2]):
            return 1.0
        
        if mask[z, y, x] == 0:
            return 1.0
        
        # 使用距离变换
        local_size = 2 * max_radius + 1
        z_start = max(0, z - max_radius)
        z_end = min(mask.shape[0], z + max_radius + 1)
        y_start = max(0, y - max_radius)
        y_end = min(mask.shape[1], y + max_radius + 1)
        x_start = max(0, x - max_radius)
        x_end = min(mask.shape[2], x + max_radius + 1)
        
        local_mask = mask[z_start:z_end, y_start:y_end, x_start:x_end]
        
        if local_mask.sum() == 0:
            return 1.0
        
        # 距离变换
        dist_transform = ndimage.distance_transform_edt(local_mask)
        
        # 找到当前点在局部区域中的位置
        local_z = z - z_start
        local_y = y - y_start
        local_x = x - x_start
        
        if (0 <= local_z < dist_transform.shape[0] and 
            0 <= local_y < dist_transform.shape[1] and 
            0 <= local_x < dist_transform.shape[2]):
            radius = dist_transform[local_z, local_y, local_x]
        else:
            radius = 1.0
        
        return max(1.0, radius)
    
    def _compute_geometric_features(self, coords: np.ndarray, radii: np.ndarray) -> np.ndarray:
        """计算几何特征 (54维)"""
        n_points = len(coords)
        features = []
        
        for i in range(n_points):
            point_features = []
            
            # 1. 坐标 (3维)
            point_features.extend(coords[i])
            
            # 2. 半径 (1维)
            point_features.append(radii[i])
            
            # 3. 方向向量 (3维)
            if i == 0 and n_points > 1:
                direction = coords[i + 1] - coords[i]
            elif i == n_points - 1 and n_points > 1:
                direction = coords[i] - coords[i - 1]
            elif n_points > 2:
                direction = coords[i + 1] - coords[i - 1]
            else:
                direction = np.array([0, 0, 1])
            
            direction = direction / (np.linalg.norm(direction) + 1e-8)
            point_features.extend(direction)
            
            # 4. 曲率 (1维)
            curvature = self._compute_curvature(coords, i)
            point_features.append(curvature)
            
            # 5. 扭转 (1维)
            torsion = self._compute_torsion(coords, i)
            point_features.append(torsion)
            
            # 6. 局部统计特征 (6维: 前后各3个点的半径统计)
            local_radii_stats = self._compute_local_radii_stats(radii, i)
            point_features.extend(local_radii_stats)
            
            # 7. 距离特征 (6维: 到前后节点的距离等)
            distance_features = self._compute_distance_features(coords, i)
            point_features.extend(distance_features)
            
            # 8. 血管分叉特征 (3维: 是否为分叉点等)
            branch_features = self._compute_branch_features(coords, i)
            point_features.extend(branch_features)
            
            # 9. 位置编码 (30维: 相对位置等复杂特征)
            position_encoding = self._compute_position_encoding(coords, i, n_points)
            point_features.extend(position_encoding)
            
            # 确保特征维度为54
            while len(point_features) < 54:
                point_features.append(0.0)
            
            features.append(point_features[:54])
        
        return np.array(features)
    
    def _compute_curvature(self, coords: np.ndarray, i: int) -> float:
        """计算曲率"""
        n = len(coords)
        if n < 3 or i == 0 or i == n - 1:
            return 0.0
        
        p1 = coords[i - 1]
        p2 = coords[i]
        p3 = coords[i + 1]
        
        v1 = p2 - p1
        v2 = p3 - p2
        
        cross_product = np.cross(v1, v2)
        cross_norm = np.linalg.norm(cross_product)
        v1_norm = np.linalg.norm(v1)
        
        if v1_norm == 0:
            return 0.0
        
        curvature = cross_norm / (v1_norm ** 3)
        return curvature
    
    def _compute_torsion(self, coords: np.ndarray, i: int) -> float:
        """计算扭转"""
        n = len(coords)
        if n < 4 or i < 2 or i >= n - 1:
            return 0.0
        
        p0 = coords[i - 2]
        p1 = coords[i - 1]
        p2 = coords[i]
        p3 = coords[i + 1]
        
        v1 = p1 - p0
        v2 = p2 - p1
        v3 = p3 - p2
        
        cross1 = np.cross(v1, v2)
        cross2 = np.cross(v2, v3)
        
        if np.linalg.norm(cross1) == 0 or np.linalg.norm(cross2) == 0:
            return 0.0
        
        torsion = np.dot(cross1, cross2) / (np.linalg.norm(cross1) * np.linalg.norm(cross2))
        return torsion
    
    def _compute_local_radii_stats(self, radii: np.ndarray, i: int) -> List[float]:
        """计算局部半径统计特征"""
        n = len(radii)
        window = 3
        
        start = max(0, i - window)
        end = min(n, i + window + 1)
        
        local_radii = radii[start:end]
        
        stats = [
            np.mean(local_radii),
            np.std(local_radii),
            np.min(local_radii),
            np.max(local_radii),
            np.median(local_radii),
            radii[i] / (np.mean(local_radii) + 1e-8)  # 相对半径
        ]
        
        return stats
    
    def _compute_distance_features(self, coords: np.ndarray, i: int) -> List[float]:
        """计算距离特征"""
        n = len(coords)
        features = []
        
        # 到前一个点的距离
        if i > 0:
            dist_prev = np.linalg.norm(coords[i] - coords[i - 1])
        else:
            dist_prev = 0.0
        features.append(dist_prev)
        
        # 到后一个点的距离
        if i < n - 1:
            dist_next = np.linalg.norm(coords[i + 1] - coords[i])
        else:
            dist_next = 0.0
        features.append(dist_next)
        
        # 到起始点的累积距离
        cum_dist = 0.0
        for j in range(i):
            if j > 0:
                cum_dist += np.linalg.norm(coords[j] - coords[j - 1])
        features.append(cum_dist)
        
        # 到终点的剩余距离
        remain_dist = 0.0
        for j in range(i, n - 1):
            remain_dist += np.linalg.norm(coords[j + 1] - coords[j])
        features.append(remain_dist)
        
        # 相对位置 (0-1)
        total_length = cum_dist + remain_dist
        relative_pos = cum_dist / (total_length + 1e-8)
        features.append(relative_pos)
        
        # 到中心的距离
        center = np.mean(coords, axis=0)
        dist_to_center = np.linalg.norm(coords[i] - center)
        features.append(dist_to_center)
        
        return features
    
    def _compute_branch_features(self, coords: np.ndarray, i: int) -> List[float]:
        """计算分叉特征"""
        # 简化版本的分叉检测
        features = [
            float(i == 0),  # 是否为起始点
            float(i == len(coords) - 1),  # 是否为终点
            0.0  # 分叉强度 (暂时设为0)
        ]
        return features
    
    def _compute_position_encoding(self, coords: np.ndarray, i: int, total_length: int) -> List[float]:
        """计算位置编码"""
        features = []
        
        # 添加正弦/余弦位置编码 (类似Transformer)
        pos = i / max(1, total_length - 1)
        
        for k in range(15):  # 15个频率
            freq = 2 ** k
            features.append(np.sin(pos * freq))
            features.append(np.cos(pos * freq))
        
        return features
    
    def _build_vessel_graph(self, centerlines: Dict, ct_array: np.ndarray, ct_image) -> Dict:
        """构建血管图"""
        all_nodes = []
        all_edges = []
        node_features = []
        node_positions = []
        node_classes = []
        node_to_vessel = []
        
        node_id = 0
        vessel_node_ranges = {}
        
        # 添加节点
        for vessel_name, vessel_data in centerlines.items():
            coords = vessel_data['coords']
            features = vessel_data['features']
            class_id = vessel_data['class_id']
            
            start_node_id = node_id
            
            for i, (coord, feature) in enumerate(zip(coords, features)):
                all_nodes.append(node_id)
                node_features.append(feature)
                node_positions.append(coord)
                node_classes.append(class_id)
                node_to_vessel.append(vessel_name)
                
                node_id += 1
            
            end_node_id = node_id - 1
            vessel_node_ranges[vessel_name] = (start_node_id, end_node_id)
            
            # 添加血管内部的边（序列连接）
            for i in range(start_node_id, end_node_id):
                all_edges.append([i, i + 1])
        
        # 添加血管间的连接（基于解剖学先验知识）
        vessel_connections = self._get_anatomical_connections()
        
        for vessel1, vessel2 in vessel_connections:
            if vessel1 in vessel_node_ranges and vessel2 in vessel_node_ranges:
                # 连接两个血管的最近节点
                range1 = vessel_node_ranges[vessel1]
                range2 = vessel_node_ranges[vessel2]
                
                # 简单策略：连接每个血管的端点
                # 可以改进为更复杂的连接策略
                all_edges.append([range1[1], range2[0]])  # vessel1的终点连接vessel2的起点
        
        # 转换为numpy数组
        node_features = np.array(node_features)
        node_positions = np.array(node_positions)
        node_classes = np.array(node_classes)
        
        if len(all_edges) > 0:
            edge_index = np.array(all_edges).T
        else:
            edge_index = np.array([[], []])
        
        return {
            'nodes': all_nodes,
            'node_features': node_features,
            'node_positions': node_positions,
            'node_classes': node_classes,
            'edge_index': edge_index,
            'vessel_node_ranges': vessel_node_ranges,
            'node_to_vessel': node_to_vessel
        }
    
    def _get_anatomical_connections(self) -> List[Tuple[str, str]]:
        """获取解剖学连接关系"""
        connections = [
            # 主要连接
            ('MPA', 'LPA'),
            ('MPA', 'RPA'),
            
            # 左侧连接
            ('LPA', 'Linternal'),
            ('LPA', 'Lupper'),
            ('Linternal', 'Lmedium'),
            ('Linternal', 'Ldown'),
            ('Lupper', 'L1+2'),
            ('Lupper', 'L1+3'),
            
            # 右侧连接
            ('RPA', 'Rinternal'),
            ('RPA', 'Rupper'),
            ('Rinternal', 'Rmedium'),
            ('Rinternal', 'RDown'),
            ('Rupper', 'R1+2'),
            ('Rupper', 'R1+3'),
        ]
        return connections
    
    def _sample_image_cubes(self, nodes: List, ct_array: np.ndarray) -> np.ndarray:
        """为每个节点采样图像块"""
        node_positions = self.vessel_graph['node_positions'] if hasattr(self, 'vessel_graph') else []
        
        if len(node_positions) == 0:
            return np.array([])
        
        image_cubes = []
        half_size = self.cube_size // 2
        
        for pos in node_positions:
            z, y, x = pos.astype(int)
            
            # 计算采样范围
            z_start = max(0, z - half_size)
            z_end = min(ct_array.shape[0], z + half_size)
            y_start = max(0, y - half_size)
            y_end = min(ct_array.shape[1], y + half_size)
            x_start = max(0, x - half_size)
            x_end = min(ct_array.shape[2], x + half_size)
            
            # 提取图像块
            cube = ct_array[z_start:z_end, y_start:y_end, x_start:x_end]
            
            # 填充到固定大小
            padded_cube = np.zeros((self.cube_size, self.cube_size, self.cube_size))
            
            actual_z = z_end - z_start
            actual_y = y_end - y_start
            actual_x = x_end - x_start
            
            padded_cube[:actual_z, :actual_y, :actual_x] = cube
            
            # 标准化
            cube_mean = np.mean(padded_cube)
            cube_std = np.std(padded_cube)
            if cube_std > 0:
                padded_cube = (padded_cube - cube_mean) / cube_std
            
            image_cubes.append(padded_cube)
        
        return np.array(image_cubes)
    
    def _prepare_training_data(self, vessel_graph: Dict, image_cubes: np.ndarray, case_id: str) -> Dict:
        """准备训练数据"""
        self.vessel_graph = vessel_graph  # 存储为实例变量以供_sample_image_cubes使用
        
        # 重新采样图像块（现在vessel_graph已设置）
        image_cubes = self._sample_image_cubes(vessel_graph['nodes'], self.ct_array if hasattr(self, 'ct_array') else np.array([]))
        
        training_data = {
            'case_id': case_id,
            'node_features': vessel_graph['node_features'],  # [N, 54]
            'node_positions': vessel_graph['node_positions'],  # [N, 3]
            'edge_index': vessel_graph['edge_index'],  # [2, E]
            'image_cubes': image_cubes,  # [N, cube_size, cube_size, cube_size]
            'node_classes': vessel_graph['node_classes'],  # [N]
            'vessel_node_ranges': vessel_graph['vessel_node_ranges'],
            'node_to_vessel': vessel_graph['node_to_vessel']
        }
        
        # 保存数据
        output_path = os.path.join(self.output_dir, f"{case_id}_processed.npz")
        np.savez_compressed(output_path, **training_data)
        
        print(f"Saved processed data to {output_path}")
        print(f"  - Nodes: {len(vessel_graph['nodes'])}")
        print(f"  - Edges: {vessel_graph['edge_index'].shape[1] if vessel_graph['edge_index'].size > 0 else 0}")
        print(f"  - Vessels: {len(vessel_graph['vessel_node_ranges'])}")
        
        return training_data
    
    def process_all_cases(self) -> List[Dict]:
        """处理所有病例"""
        # 获取所有病例ID
        ct_files = [f for f in os.listdir(self.ct_dir) if f.endswith('.nii')]
        case_ids = [f.replace('.nii', '') for f in ct_files]
        
        print(f"Found {len(case_ids)} cases to process")
        
        processed_cases = []
        for case_id in case_ids:
            try:
                # 存储ct_array供_sample_image_cubes使用
                ct_path = os.path.join(self.ct_dir, f"{case_id}.nii")
                ct_image = sitk.ReadImage(ct_path)
                self.ct_array = sitk.GetArrayFromImage(ct_image)
                
                result = self.process_case(case_id)
                if result:
                    processed_cases.append(result)
            except Exception as e:
                print(f"Error processing case {case_id}: {e}")
                continue
        
        print(f"Successfully processed {len(processed_cases)} cases")
        return processed_cases

def main():
    """主函数"""
    preprocessor = VesselPreprocessor(
        ct_dir="train",
        label_dir="label_filtered", 
        output_dir="processed_data",
        cube_size=32
    )
    
    # 处理所有病例
    results = preprocessor.process_all_cases()
    
    print(f"\n🎉 Processing completed!")
    print(f"📊 Total processed cases: {len(results)}")
    print(f"📁 Output directory: processed_data/")

if __name__ == "__main__":
    main()
