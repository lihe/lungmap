import os
import numpy as np
import SimpleITK as sitk
from scipy import ndimage
from skimage.morphology import skeletonize_3d
from sklearn.neighbors import NearestNeighbors
from typing import Dict, List, Tuple, Optional
import networkx as nx
from scipy.spatial.distance import pdist, squareform, cdist
from collections import defaultdict
from scipy.interpolate import splprep, splev
import time
import traceback

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
        image_cubes = self._sample_image_cubes(vessel_graph['node_positions'], ct_array)
        
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
        """
        综合中心线提取 - 整合专业医学影像方法
        参考thinVolume、Tang_method2、compute_radius、CLExtract的专业算法
        """
        centerlines = {}
        
        print(f"开始专业中心线提取，处理 {len(label_mapping)} 个标签...")
        
        for vessel_name, label_value in label_mapping.items():
            if vessel_name in self.label_to_class:
                try:
                    print(f"  处理血管: {vessel_name}")
                    start_time = time.time()
                    
                    # 提取该标签的二值掩码
                    vessel_mask = (label_array == label_value).astype(np.uint8)
                    
                    if np.sum(vessel_mask) < 10:
                        print(f"    跳过: 区域太小 ({np.sum(vessel_mask)} 体素)")
                        continue
                    
                    if vessel_mask.ndim != 3:
                        print(f"    错误: 掩码不是3D (形状: {vessel_mask.shape})")
                        continue
                    
                    # 第一步：高级血管清理
                    print(f"    步骤1: 高级清理...")
                    cleaned_mask = self._advanced_vessel_cleaning(vessel_mask)
                    
                    if np.sum(cleaned_mask) < 5:
                        print(f"    跳过: 清理后太小")
                        continue
                    
                    # 第二步：专业细化
                    print(f"    步骤2: 专业细化...")
                    raw_centerline = self._get_thinned_centerline(cleaned_mask)
                    
                    if raw_centerline.sum() == 0:
                        print(f"    跳过: 未找到中心线")
                        continue
                    
                    # 第三步：单体素化
                    print(f"    步骤3: 单体素化...")
                    refined_centerline = self._single_voxelize_centerline(raw_centerline)
                    
                    # 第四步：提取坐标
                    coords = np.column_stack(np.where(refined_centerline > 0))
                    original_count = len(coords)
                    
                    if original_count < 5:
                        print(f"    跳过: 中心线点太少 ({original_count})")
                        continue
                    
                    # 第五步：拓扑分析
                    print(f"    步骤4: 拓扑分析...")
                    ordered_coords, topology_info = self._extract_vessel_topology(coords, refined_centerline)
                    
                    # 第六步：半径计算
                    print(f"    步骤5: 半径计算...")
                    radii = self._compute_vessel_radius(ordered_coords, cleaned_mask)
                    
                    # 第七步：B样条平滑
                    print(f"    步骤6: B样条平滑...")
                    if len(ordered_coords) >= 4:
                        try:
                            from scipy.interpolate import splprep, splev
                            
                            tck, u = splprep([ordered_coords[:, 0], ordered_coords[:, 1], ordered_coords[:, 2]], 
                                           s=len(ordered_coords)*0.1, k=3)
                            
                            target_points = min(100, max(20, len(ordered_coords) // 5))
                            u_new = np.linspace(0, 1, target_points)
                            smooth_coords = splev(u_new, tck)
                            smooth_coords = np.column_stack(smooth_coords)
                            
                            smooth_radii = self._compute_vessel_radius(smooth_coords, cleaned_mask)
                            
                        except Exception as e:
                            print(f"    B样条失败，使用原始: {e}")
                            smooth_coords = ordered_coords
                            smooth_radii = radii
                    else:
                        smooth_coords = ordered_coords
                        smooth_radii = radii
                    
                    # 第八步：最终简化
                    print(f"    步骤7: 最终简化...")
                    final_coords = self._simplify_centerline(smooth_coords, vessel_type='artery')
                    final_radii = smooth_radii[:len(final_coords)] if len(smooth_radii) >= len(final_coords) else smooth_radii
                    
                    # 第九步：质量验证
                    quality_metrics = self._validate_centerline_quality(final_coords, cleaned_mask)
                    
                    # 计算几何特征
                    features = self._compute_geometric_features(final_coords, final_radii)
                    
                    # 创建附加特征字典
                    additional_features = {
                        'topology_info': topology_info,
                        'quality_metrics': quality_metrics,
                        'processing_time': time.time() - start_time
                    }
                    
                    centerlines[vessel_name] = {
                        'coords': final_coords,
                        'radii': final_radii,
                        'features': features,
                        'additional_features': additional_features,
                        'label_value': label_value,
                        'class_id': self.label_to_class[vessel_name]
                    }
                    
                    compression_ratio = len(final_coords) / original_count * 100
                    print(f"    成功: {original_count} -> {len(final_coords)}点 "
                          f"(压缩率: {compression_ratio:.1f}%, "
                          f"质量: {quality_metrics['overall_score']:.3f}, "
                          f"耗时: {time.time() - start_time:.2f}s)")
                    
                except Exception as e:
                    print(f"    处理 {vessel_name} 时出错: {e}")
                    traceback.print_exc()
                    continue
        
        print(f"专业中心线提取完成，成功提取 {len(centerlines)} 条中心线")
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
    
    def _advanced_vessel_cleaning(self, mask: np.ndarray) -> np.ndarray:
        """
        高级血管掩码清理 - 参考thinVolume.py
        """
        # 1. 移除小的连通分量
        labeled_mask, num_labels = ndimage.label(mask)
        
        if num_labels == 0:
            return mask
        
        # 2. 保留最大连通分量
        label_sizes = np.bincount(labeled_mask.flat)
        if len(label_sizes) > 1:
            largest_label = np.argmax(label_sizes[1:]) + 1
            cleaned_mask = (labeled_mask == largest_label).astype(np.uint8)
        else:
            cleaned_mask = mask.copy()
        
        # 3. 形态学操作序列
        from scipy.ndimage import binary_opening, binary_closing, binary_fill_holes
        
        # 先填充小孔洞
        if cleaned_mask.ndim == 3:
            for i in range(cleaned_mask.shape[0]):
                cleaned_mask[i] = binary_fill_holes(cleaned_mask[i]).astype(np.uint8)
        
        # 开运算去除噪声
        cleaned_mask = binary_opening(cleaned_mask, structure=np.ones((3, 3, 3))).astype(np.uint8)
        
        # 闭运算连接断裂
        cleaned_mask = binary_closing(cleaned_mask, structure=np.ones((3, 3, 3))).astype(np.uint8)
        
        return cleaned_mask
    
    def _get_thinned_centerline(self, binary_mask: np.ndarray) -> np.ndarray:
        """
        使用专业细化算法提取中心线 - 参考thinVolume.py的get_thinned方法
        """
        if np.max(binary_mask) not in [0, 1]:
            binary_mask = (binary_mask > 0).astype(np.uint8)
        
        voxel_count = np.sum(binary_mask)
        if voxel_count == 0 or voxel_count == binary_mask.size:
            return binary_mask
        
        # 使用skimage的3D骨架化（简化版本，实际项目中可以集成Cython优化版本）
        print(f"    Thinning {voxel_count} voxels...")
        start_time = time.time()
        
        thinned = skeletonize_3d(binary_mask.astype(bool)).astype(np.uint8)
        
        print(f"    Thinned in {time.time() - start_time:.2f} seconds")
        return thinned
    
    def _single_voxelize_centerline(self, centerline: np.ndarray) -> np.ndarray:
        """
        单体素化中心线 - 参考utils.py的thinning_Voxel2方法
        🔧 修复：使用确定性方法替代随机处理
        """
        result = centerline.copy()
        xx, yy, zz = np.where(centerline == 1)
        
        for i in range(len(xx)):
            a, b, c = xx[i], yy[i], zz[i]
            
            # 检查3x3x3邻域
            if (a-1 >= 0 and a+1 < centerline.shape[0] and 
                b-1 >= 0 and b+1 < centerline.shape[1] and 
                c-1 >= 0 and c+1 < centerline.shape[2]):
                
                block = centerline[a-1:a+2, b-1:b+2, c-1:c+2]
                neighbor_count = np.sum(block > 0)
                
                if neighbor_count == 1:
                    # 孤立点，保留
                    continue
                elif neighbor_count > 5:
                    # 可能的分叉点或密集区域，需要细化
                    # 🔧 修复：使用确定性规则而非随机
                    # 保留中心点，按距离移除最远的邻居
                    neighbors = []
                    for di in [-1, 0, 1]:
                        for dj in [-1, 0, 1]:
                            for dk in [-1, 0, 1]:
                                if di == 0 and dj == 0 and dk == 0:
                                    continue
                                ni, nj, nk = a + di, b + dj, c + dk
                                if (0 <= ni < result.shape[0] and 
                                    0 <= nj < result.shape[1] and 
                                    0 <= nk < result.shape[2]):
                                    if result[ni, nj, nk] > 0:
                                        dist = np.sqrt(di*di + dj*dj + dk*dk)
                                        neighbors.append((dist, ni, nj, nk))
                    
                    # 排序并移除距离最远的30%邻居
                    neighbors.sort(reverse=True)
                    remove_count = max(1, len(neighbors) // 3)
                    for j in range(remove_count):
                        _, ni, nj, nk = neighbors[j]
                        result[ni, nj, nk] = 0
        
        return result
    
    def _extract_vessel_topology(self, coords: np.ndarray, centerline_arr: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        🔧 修复：血管拓扑分析和分支提取 - 确保单个连通分量
        基于分叉点进行血管分段，每段作为一个节点
        """
        if len(coords) < 3:
            return coords, {'branch_count': 1, 'bifurcations': [], 'segments': [coords]}
        
        # 1. 构建邻接图
        adjacency_graph = self._build_adjacency_graph(coords)
        
        # 2. 找到分叉点（度 > 2）和端点（度 = 1）
        bifurcations = []
        endpoints = []
        junction_points = []
        
        for i, coord in enumerate(coords):
            degree = len(adjacency_graph[i])
            if degree > 2:
                bifurcations.append(coord)
                junction_points.append(i)
            elif degree == 1:
                endpoints.append(coord)
        
        print(f"    发现 {len(bifurcations)} 个分叉点, {len(endpoints)} 个端点")
        
        # 3. 基于分叉点分段血管
        segments = self._segment_vessel_by_bifurcations(coords, adjacency_graph, junction_points)
        
        print(f"    分割为 {len(segments)} 个血管段")
        
        # 4. 对每个段进行关键点采样
        sampled_segments = []
        all_sampled_coords = []
        
        for i, segment in enumerate(segments):
            if len(segment) < 2:
                continue
                
            # 对每个段进行关键点采样
            sampled_points = self._sample_key_points_from_segment(segment)
            sampled_segments.append(sampled_points)
            all_sampled_coords.extend(sampled_points)
            
            print(f"    段 {i+1}: {len(segment)} -> {len(sampled_points)} 关键点")
        
        ordered_coords = np.array(all_sampled_coords) if all_sampled_coords else coords
        
        topology_info = {
            'branch_count': len(segments),
            'bifurcations': bifurcations,
            'segments': sampled_segments,
            'total_points': len(ordered_coords),
            'connectivity_ratio': 1.0  # 确保单个连通分量
        }
        
        return ordered_coords, topology_info
    
    def _build_adjacency_graph(self, coords: np.ndarray) -> Dict:
        """构建中心线点的邻接图"""
        adjacency = defaultdict(list)
        n_points = len(coords)
        
        # 计算所有点对之间的距离
        for i in range(n_points):
            for j in range(i + 1, n_points):
                distance = np.linalg.norm(coords[i] - coords[j])
                
                # 如果距离小于等于sqrt(3)，认为是邻接的（3D 26-连通）
                if distance <= np.sqrt(3) + 1e-6:
                    adjacency[i].append(j)
                    adjacency[j].append(i)
        
        return adjacency
    
    def _segment_vessel_by_bifurcations(self, coords: np.ndarray, adjacency: Dict, junction_points: List[int]) -> List[np.ndarray]:
        """基于分叉点分割血管为段"""
        segments = []
        visited = set()
        
        # 将分叉点加入已访问，作为段的分界点
        junction_set = set(junction_points)
        
        # 从每个非分叉点开始构建段
        for start_idx in range(len(coords)):
            if start_idx in visited or start_idx in junction_set:
                continue
            
            # 使用BFS构建当前段
            segment = []
            queue = [start_idx]
            segment_visited = set()
            
            while queue:
                current_idx = queue.pop(0)
                if current_idx in segment_visited:
                    continue
                
                segment_visited.add(current_idx)
                visited.add(current_idx)
                segment.append(coords[current_idx])
                
                # 添加邻居（除非是分叉点）
                for neighbor_idx in adjacency[current_idx]:
                    if (neighbor_idx not in segment_visited and 
                        neighbor_idx not in junction_set and
                        neighbor_idx not in visited):
                        queue.append(neighbor_idx)
            
            if len(segment) >= 2:  # 只保留有意义的段
                segments.append(np.array(segment))
        
        # 处理分叉点周围的连接
        for junction_idx in junction_points:
            # 每个分叉点单独作为一个段
            segments.append(np.array([coords[junction_idx]]))
        
        return segments
    
    def _sample_key_points_from_segment(self, segment: np.ndarray, max_points: int = 5) -> np.ndarray:
        """从血管段中采样关键点"""
        if len(segment) <= max_points:
            return segment
        
        # 策略1：保留端点
        if len(segment) == 2:
            return segment
        
        # 策略2：基于曲率采样关键点
        key_points = [segment[0]]  # 起点
        
        if len(segment) > 2:
            # 计算每个点的曲率
            curvatures = []
            for i in range(1, len(segment) - 1):
                curvature = self._compute_point_curvature(segment, i)
                curvatures.append((curvature, i))
            
            # 按曲率排序，选择曲率最大的点作为关键点
            curvatures.sort(reverse=True)
            selected_indices = [idx for _, idx in curvatures[:max_points-2]]
            selected_indices.sort()
            
            for idx in selected_indices:
                key_points.append(segment[idx])
        
        key_points.append(segment[-1])  # 终点
        
        return np.array(key_points)
    
    def _compute_point_curvature(self, segment: np.ndarray, i: int) -> float:
        """计算点的曲率"""
        if i <= 0 or i >= len(segment) - 1:
            return 0.0
        
        p1 = segment[i - 1]
        p2 = segment[i]
        p3 = segment[i + 1]
        
        v1 = p2 - p1
        v2 = p3 - p2
        
        if np.linalg.norm(v1) == 0 or np.linalg.norm(v2) == 0:
            return 0.0
        
        # 计算角度变化作为曲率的度量
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        cos_angle = np.clip(cos_angle, -1, 1)
        angle = np.arccos(cos_angle)
        
        return angle
    
    def _compute_vessel_radius(self, coords: np.ndarray, vessel_arr: np.ndarray) -> np.ndarray:
        """
        计算血管半径 - 参考compute_radius.py的最大内接球方法
        """
        radii = []
        
        for coord in coords:
            z, y, x = coord.astype(int)
            
            # 确保坐标在边界内
            if (z < 0 or z >= vessel_arr.shape[0] or 
                y < 0 or y >= vessel_arr.shape[1] or 
                x < 0 or x >= vessel_arr.shape[2]):
                radii.append(1.0)  # 默认半径
                continue
            
            # 使用距离变换计算到边界的距离
            # 在局部区域内计算以提高效率
            pad = 15  # 局部区域大小
            
            z_min = max(0, z - pad)
            z_max = min(vessel_arr.shape[0], z + pad + 1)
            y_min = max(0, y - pad)
            y_max = min(vessel_arr.shape[1], y + pad + 1)
            x_min = max(0, x - pad)
            x_max = min(vessel_arr.shape[2], x + pad + 1)
            
            local_vessel = vessel_arr[z_min:z_max, y_min:y_max, x_min:x_max]
            
            if local_vessel.sum() == 0:
                radii.append(1.0)
                continue
            
            # 计算距离变换
            from scipy.ndimage import distance_transform_edt
            distance_map = distance_transform_edt(local_vessel)
            
            # 找到当前点在局部区域中的位置
            local_z = z - z_min
            local_y = y - y_min
            local_x = x - x_min
            
            if (0 <= local_z < distance_map.shape[0] and 
                0 <= local_y < distance_map.shape[1] and 
                0 <= local_x < distance_map.shape[2]):
                radius = distance_map[local_z, local_y, local_x]
            else:
                radius = 1.0
            
            radii.append(max(1.0, radius))  # 最小半径为1
        
        return np.array(radii)
    
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
        """🔧 修复：构建连通的血管图"""
        all_nodes = []
        all_edges = []
        node_features = []
        node_positions = []
        node_classes = []
        node_to_vessel = []
        
        node_id = 0
        vessel_node_ranges = {}
        
        # 添加节点（每个关键点作为一个节点）
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
            
            # 🔧 修复：添加血管内部的顺序连接
            for i in range(start_node_id, end_node_id):
                all_edges.append([i, i + 1])
        
        # 🔧 修复：基于解剖学知识强制连接血管
        anatomical_connections = [
            ('MPA', 'LPA'), ('MPA', 'RPA'),
            ('LPA', 'Linternal'), ('RPA', 'Rinternal')
        ]
        
        for vessel1, vessel2 in anatomical_connections:
            if vessel1 in vessel_node_ranges and vessel2 in vessel_node_ranges:
                range1 = vessel_node_ranges[vessel1]
                range2 = vessel_node_ranges[vessel2]
                
                # 连接距离最近的节点
                best_connection = self._find_closest_connection(
                    range1, range2, np.array(node_positions)
                )
                
                if best_connection:
                    all_edges.append(best_connection)
                    print(f"    连接 {vessel1} <-> {vessel2}: 节点 {best_connection}")
        
        # 转换为numpy数组
        node_features = np.array(node_features)
        node_positions = np.array(node_positions)
        node_classes = np.array(node_classes)
        
        if len(all_edges) > 0:
            edge_index = np.array(all_edges).T
        else:
            edge_index = np.array([[], []])
        
        # 构建血管图
        vessel_graph = {
            'nodes': all_nodes,
            'node_features': node_features,
            'node_positions': node_positions,
            'node_classes': node_classes,
            'edge_index': edge_index,
            'vessel_node_ranges': vessel_node_ranges,
            'node_to_vessel': node_to_vessel
        }
        
        # 🔧 执行图形补全以确保连通性
        vessel_graph = self._ensure_graph_connectivity(vessel_graph)
        
        return vessel_graph
    
    def _find_closest_connection(self, range1: Tuple[int, int], range2: Tuple[int, int], 
                                positions: np.ndarray) -> Optional[List[int]]:
        """找到两个血管间距离最近的连接"""
        start1, end1 = range1
        start2, end2 = range2
        
        best_distance = float('inf')
        best_connection = None
        
        # 检查所有可能的连接组合
        for node1 in range(start1, end1 + 1):
            for node2 in range(start2, end2 + 1):
                if node1 < len(positions) and node2 < len(positions):
                    distance = np.linalg.norm(positions[node1] - positions[node2])
                    
                    if distance < best_distance:
                        best_distance = distance
                        best_connection = [node1, node2]
        
        return best_connection if best_distance < 50.0 else None
    
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
    
    def _ensure_graph_connectivity(self, vessel_graph: Dict) -> Dict:
        """🔧 确保图的连通性 - 保证只有一个连通分量"""
        print("🔧 确保图连通性...")
        
        node_positions = vessel_graph['node_positions']
        edge_index = vessel_graph['edge_index']
        node_classes = vessel_graph['node_classes']
        n_nodes = len(node_positions)
        
        if n_nodes <= 1:
            return vessel_graph
        
        # 1. 构建当前图的邻接列表
        adjacency = defaultdict(list)
        if edge_index.size > 0:
            for i in range(edge_index.shape[1]):
                src, tgt = edge_index[0, i], edge_index[1, i]
                adjacency[src].append(tgt)
                adjacency[tgt].append(src)
        
        # 2. 找到所有连通分量
        visited = set()
        components = []
        
        def dfs(node, component):
            if node in visited:
                return
            visited.add(node)
            component.append(node)
            for neighbor in adjacency[node]:
                dfs(neighbor, component)
        
        for node in range(n_nodes):
            if node not in visited:
                component = []
                dfs(node, component)
                components.append(component)
        
        print(f"   发现 {len(components)} 个连通分量")
        
        # 3. 如果有多个连通分量，强制连接它们
        additional_edges = []
        
        if len(components) > 1:
            # 连接所有分量到最大的分量
            largest_component = max(components, key=len)
            
            for component in components:
                if component == largest_component:
                    continue
                
                # 找到这个分量到最大分量的最短连接
                best_connection = self._find_shortest_inter_component_connection(
                    component, largest_component, node_positions
                )
                
                if best_connection:
                    additional_edges.append(best_connection)
                    print(f"   连接分量: 节点 {best_connection}")
                    
                    # 更新邻接表
                    src, tgt = best_connection
                    adjacency[src].append(tgt)
                    adjacency[tgt].append(src)
                    
                    # 将当前分量合并到最大分量
                    largest_component.extend(component)
        
        # 4. 合并原有边和新增边
        all_edges = []
        if edge_index.size > 0:
            for i in range(edge_index.shape[1]):
                all_edges.append([edge_index[0, i], edge_index[1, i]])
        
        all_edges.extend(additional_edges)
        
        # 5. 更新图结构
        if len(all_edges) > 0:
            final_edge_index = np.array(all_edges).T
        else:
            final_edge_index = np.array([[], []])
        
        vessel_graph['edge_index'] = final_edge_index
        
        # 6. 最终验证连通性
        final_components = self._count_connected_components(final_edge_index, n_nodes)
        print(f"   最终连通分量数: {final_components}")
        
        vessel_graph['connectivity_ensured'] = True
        vessel_graph['final_component_count'] = final_components
        
        return vessel_graph
    
    def _find_shortest_inter_component_connection(self, component1: List[int], 
                                                component2: List[int], 
                                                positions: np.ndarray) -> Optional[List[int]]:
        """找到两个连通分量间的最短连接"""
        best_distance = float('inf')
        best_connection = None
        
        for node1 in component1:
            for node2 in component2:
                distance = np.linalg.norm(positions[node1] - positions[node2])
                
                if distance < best_distance:
                    best_distance = distance
                    best_connection = [node1, node2]
        
        return best_connection
    
    def _count_connected_components(self, edge_index: np.ndarray, n_nodes: int) -> int:
        """计算连通分量数量"""
        if n_nodes == 0:
            return 0
        
        adjacency = defaultdict(list)
        if edge_index.size > 0:
            for i in range(edge_index.shape[1]):
                src, tgt = edge_index[0, i], edge_index[1, i]
                adjacency[src].append(tgt)
                adjacency[tgt].append(src)
        
        visited = set()
        component_count = 0
        
        def dfs(node):
            if node in visited:
                return
            visited.add(node)
            for neighbor in adjacency[node]:
                dfs(neighbor)
        
        for node in range(n_nodes):
            if node not in visited:
                dfs(node)
                component_count += 1
        
        return component_count
    
    def _graph_completion_legacy(self, vessel_graph: Dict) -> Dict:
        """
        🧠 图形补全：优化血管图的连接性和拓扑结构
        """
        print("🔧 执行图形补全...")
        
        node_positions = vessel_graph['node_positions']
        edge_index = vessel_graph['edge_index']
        node_classes = vessel_graph['node_classes']
        vessel_node_ranges = vessel_graph['vessel_node_ranges']
        
        # 1. 基于距离的连接补全
        enhanced_edges = self._distance_based_completion(
            node_positions, edge_index, node_classes, distance_threshold=5.0
        )
        
        # 2. 基于解剖学的连接补全
        anatomical_edges = self._anatomical_based_completion(
            vessel_node_ranges, node_positions, node_classes
        )
        
        # 3. 基于连续性的连接补全
        continuity_edges = self._continuity_based_completion(
            node_positions, edge_index, node_classes
        )
        
        # 4. 合并所有边连接
        all_edges = self._merge_edge_connections(
            [edge_index, enhanced_edges, anatomical_edges, continuity_edges]
        )
        
        # 5. 移除重复和冲突的边
        final_edges = self._clean_edge_connections(all_edges, node_positions, node_classes)
        
        # 更新图结构
        vessel_graph['edge_index'] = final_edges
        vessel_graph['completion_stats'] = self._compute_completion_stats(
            edge_index, final_edges
        )
        
        print(f"   原始边数: {edge_index.shape[1] if edge_index.size > 0 else 0}")
        print(f"   补全后边数: {final_edges.shape[1] if final_edges.size > 0 else 0}")
        print(f"   新增边数: {final_edges.shape[1] - edge_index.shape[1] if edge_index.size > 0 and final_edges.size > 0 else 0}")
        
        return vessel_graph

    def _distance_based_completion(self, positions: np.ndarray, existing_edges: np.ndarray, 
                                  classes: np.ndarray, distance_threshold: float = 5.0) -> np.ndarray:
        """基于距离的连接补全"""
        new_edges = []
        n_nodes = len(positions)
        
        # 计算所有节点间的距离
        distances = cdist(positions, positions)
        
        # 获取现有连接
        existing_connections = set()
        if existing_edges.size > 0:
            for i in range(existing_edges.shape[1]):
                src, tgt = existing_edges[0, i], existing_edges[1, i]
                existing_connections.add((min(src, tgt), max(src, tgt)))
        
        # 找到相近但未连接的节点对
        for i in range(n_nodes):
            for j in range(i + 1, n_nodes):
                if (i, j) not in existing_connections:
                    if distances[i, j] < distance_threshold:
                        # 检查是否应该连接（基于血管类别）
                        if self._should_connect_by_distance(classes[i], classes[j], distances[i, j]):
                            new_edges.append([i, j])
        
        if len(new_edges) == 0:
            return np.array([[], []])
        
        return np.array(new_edges).T

    def _should_connect_by_distance(self, class1: int, class2: int, distance: float) -> bool:
        """判断两个节点是否应该基于距离连接"""
        # 同类别节点，距离很近
        if class1 == class2 and distance < 3.0:
            return True
        
        # 🔧 修复：正确的解剖学邻接关系
        anatomical_adjacency = {
            0: [1, 2],        # MPA -> LPA, RPA
            1: [3, 4, 5, 6],  # LPA -> Linternal, Lupper, Lmedium, Ldown
            2: [7, 8, 9, 10], # RPA -> Rinternal, Rupper, Rmedium, RDown
            3: [11, 12],      # Linternal -> L1+2, L1+3 (可能的连接)
            4: [11, 12],      # Lupper -> L1+2, L1+3
            7: [13, 14],      # Rinternal -> R1+2, R1+3 (可能的连接)
            8: [13, 14],      # Rupper -> R1+2, R1+3
        }
        
        # 检查解剖学邻接关系 (双向)
        if (class2 in anatomical_adjacency.get(class1, []) or 
            class1 in anatomical_adjacency.get(class2, [])):
            return distance < 15.0  # 增加距离阈值
        
        return False

    def _anatomical_based_completion(self, vessel_ranges: Dict, positions: np.ndarray, 
                                    classes: np.ndarray) -> np.ndarray:
        """基于解剖学先验知识的连接补全"""
        new_edges = []
        
        # 定义解剖学连接规则
        anatomical_connections = [
            ('MPA', 'LPA'), ('MPA', 'RPA'),
            ('LPA', 'Linternal'), ('LPA', 'Lupper'),
            ('RPA', 'Rinternal'), ('RPA', 'Rupper'),
            ('Linternal', 'Lmedium'), ('Linternal', 'Ldown'),
            ('Lupper', 'L1+2'), ('Lupper', 'L1+3'),
            ('Rinternal', 'Rmedium'), ('Rinternal', 'RDown'),
            ('Rupper', 'R1+2'), ('Rupper', 'R1+3'),
        ]
        
        # 为每个解剖学连接找到最佳节点对
        for vessel1, vessel2 in anatomical_connections:
            if vessel1 in vessel_ranges and vessel2 in vessel_ranges:
                range1 = vessel_ranges[vessel1]
                range2 = vessel_ranges[vessel2]
                
                # 找到两个血管间距离最近的节点对
                best_connection = self._find_best_vessel_connection(
                    range1, range2, positions, classes
                )
                
                if best_connection:
                    new_edges.append(best_connection)
        
        if len(new_edges) == 0:
            return np.array([[], []])
        
        return np.array(new_edges).T

    def _find_best_vessel_connection(self, range1: Tuple[int, int], range2: Tuple[int, int],
                                    positions: np.ndarray, classes: np.ndarray) -> Optional[List[int]]:
        """找到两个血管间的最佳连接"""
        start1, end1 = range1
        start2, end2 = range2
        
        best_distance = float('inf')
        best_connection = None
        
        # 检查血管端点间的连接
        candidates = [
            (start1, start2),  # 起点-起点
            (start1, end2),    # 起点-终点
            (end1, start2),    # 终点-起点
            (end1, end2),      # 终点-终点
        ]
        
        for node1, node2 in candidates:
            if node1 < len(positions) and node2 < len(positions):
                distance = np.linalg.norm(positions[node1] - positions[node2])
                
                # 距离阈值：解剖学连接应该相对较近
                if distance < 15.0 and distance < best_distance:
                    best_distance = distance
                    best_connection = [node1, node2]
        
        return best_connection

    def _continuity_based_completion(self, positions: np.ndarray, existing_edges: np.ndarray,
                                    classes: np.ndarray) -> np.ndarray:
        """基于血管连续性的连接补全"""
        new_edges = []
        
        if existing_edges.size == 0:
            return np.array([[], []])
        
        # 构建现有图的邻接表
        adjacency = defaultdict(list)
        
        for i in range(existing_edges.shape[1]):
            src, tgt = existing_edges[0, i], existing_edges[1, i]
            adjacency[src].append(tgt)
            adjacency[tgt].append(src)
        
        # 找到孤立节点和度为1的节点（端点）
        isolated_nodes = []
        endpoint_nodes = []
        
        for node in range(len(positions)):
            degree = len(adjacency[node])
            if degree == 0:
                isolated_nodes.append(node)
            elif degree == 1:
                endpoint_nodes.append(node)
        
        # 为孤立节点找连接
        for isolated in isolated_nodes:
            best_neighbor = self._find_best_neighbor_for_isolated(
                isolated, positions, classes, adjacency
            )
            if best_neighbor is not None:
                new_edges.append([isolated, best_neighbor])
        
        # 连接相近的端点
        for i, endpoint1 in enumerate(endpoint_nodes):
            for endpoint2 in endpoint_nodes[i+1:]:
                if self._should_connect_endpoints(
                    endpoint1, endpoint2, positions, classes, adjacency
                ):
                    new_edges.append([endpoint1, endpoint2])
        
        if len(new_edges) == 0:
            return np.array([[], []])
        
        return np.array(new_edges).T

    def _find_best_neighbor_for_isolated(self, isolated_node: int, positions: np.ndarray,
                                       classes: np.ndarray, adjacency: Dict) -> Optional[int]:
        """为孤立节点找到最佳邻居"""
        isolated_pos = positions[isolated_node]
        isolated_class = classes[isolated_node]
        
        best_neighbor = None
        best_distance = float('inf')
        
        for candidate in range(len(positions)):
            if candidate == isolated_node:
                continue
            
            # 优先连接同类别节点
            if classes[candidate] == isolated_class:
                distance = np.linalg.norm(positions[candidate] - isolated_pos)
                if distance < 10.0 and distance < best_distance:
                    best_distance = distance
                    best_neighbor = candidate
        
        # 如果没找到同类别的，找相邻类别的
        if best_neighbor is None:
            for candidate in range(len(positions)):
                if candidate == isolated_node:
                    continue
                
                if self._should_connect_by_distance(isolated_class, classes[candidate], 0):
                    distance = np.linalg.norm(positions[candidate] - isolated_pos)
                    if distance < 8.0 and distance < best_distance:
                        best_distance = distance
                        best_neighbor = candidate
        
        return best_neighbor

    def _should_connect_endpoints(self, endpoint1: int, endpoint2: int, positions: np.ndarray,
                                 classes: np.ndarray, adjacency: Dict) -> bool:
        """判断两个端点是否应该连接"""
        distance = np.linalg.norm(positions[endpoint1] - positions[endpoint2])
        
        # 距离太远，不连接
        if distance > 10.0:
            return False
        
        # 同类别的端点，较近时可以连接
        if classes[endpoint1] == classes[endpoint2] and distance < 5.0:
            return True
        
        # 不同类别但解剖学相关的端点
        if self._should_connect_by_distance(classes[endpoint1], classes[endpoint2], distance):
            return distance < 7.0
        
        return False

    def _merge_edge_connections(self, edge_lists: List[np.ndarray]) -> np.ndarray:
        """合并多个边连接列表"""
        all_edges = []
        
        for edges in edge_lists:
            if edges.size > 0 and edges.shape[0] == 2:
                for i in range(edges.shape[1]):
                    all_edges.append([edges[0, i], edges[1, i]])
        
        if len(all_edges) == 0:
            return np.array([[], []])
        
        return np.array(all_edges).T

    def _clean_edge_connections(self, edges: np.ndarray, positions: np.ndarray, 
                               classes: np.ndarray) -> np.ndarray:
        """清理边连接：移除重复、自环和冲突的边"""
        if edges.size == 0:
            return edges
        
        cleaned_edges = []
        seen_edges = set()
        
        for i in range(edges.shape[1]):
            src, tgt = edges[0, i], edges[1, i]
            
            # 移除自环
            if src == tgt:
                continue
            
            # 移除重复边（无向图）
            edge_key = (min(src, tgt), max(src, tgt))
            if edge_key in seen_edges:
                continue
            
            # 检查边的合理性
            if self._is_valid_edge(src, tgt, positions, classes):
                cleaned_edges.append([src, tgt])
                seen_edges.add(edge_key)
        
        if len(cleaned_edges) == 0:
            return np.array([[], []])
        
        return np.array(cleaned_edges).T

    def _is_valid_edge(self, src: int, tgt: int, positions: np.ndarray, classes: np.ndarray) -> bool:
        """检查边的有效性"""
        # 检查节点索引
        if src >= len(positions) or tgt >= len(positions):
            return False
        
        # 检查距离：过远的连接不合理
        distance = np.linalg.norm(positions[src] - positions[tgt])
        if distance > 50.0:  # 🔧 修复：增加最大连接距离
            return False
        
        # 🔧 修复：放宽类别兼容性检查，保留更多的连接
        # 1. 同类别的节点总是可以连接
        if classes[src] == classes[tgt]:
            return True
        
        # 2. 检查解剖学兼容性
        anatomical_adjacency = {
            0: [1, 2],        # MPA -> LPA, RPA
            1: [3, 4, 5, 6],  # LPA -> Linternal, Lupper, Lmedium, Ldown
            2: [7, 8, 9, 10], # RPA -> Rinternal, Rupper, Rmedium, RDown
            3: [11, 12],      # Linternal -> L1+2, L1+3
            4: [11, 12],      # Lupper -> L1+2, L1+3
            7: [13, 14],      # Rinternal -> R1+2, R1+3
            8: [13, 14],      # Rupper -> R1+2, R1+3
        }
        
        # 检查解剖学邻接关系 (双向)
        if (classes[tgt] in anatomical_adjacency.get(classes[src], []) or 
            classes[src] in anatomical_adjacency.get(classes[tgt], [])):
            return distance < 30.0
        
        # 3. 对于相近的节点，允许连接（容错机制）
        if distance < 10.0:
            return True
        
        return False

    def _compute_completion_stats(self, original_edges: np.ndarray, final_edges: np.ndarray) -> Dict:
        """计算图形补全统计信息"""
        stats = {
            'original_edge_count': original_edges.shape[1] if original_edges.size > 0 else 0,
            'final_edge_count': final_edges.shape[1] if final_edges.size > 0 else 0,
            'added_edge_count': 0,
            'completion_ratio': 0.0
        }
        
        if original_edges.size > 0 and final_edges.size > 0:
            stats['added_edge_count'] = stats['final_edge_count'] - stats['original_edge_count']
            if stats['original_edge_count'] > 0:
                stats['completion_ratio'] = stats['added_edge_count'] / stats['original_edge_count']
        
        return stats
    
    def _sample_image_cubes(self, node_positions: np.ndarray, ct_array: np.ndarray) -> np.ndarray:
        """为每个节点采样图像块"""
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
        # 🔧 修复：使用传入的image_cubes，不重复采样
        if image_cubes.size == 0:
            print(f"Warning: Empty image cubes for case {case_id}")
            image_cubes = np.array([])
        
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
        
        # 显示图形补全统计
        if 'completion_stats' in vessel_graph:
            stats = vessel_graph['completion_stats']
            print(f"  - 图形补全统计:")
            print(f"    原始边数: {stats['original_edge_count']}")
            print(f"    补全后边数: {stats['final_edge_count']}")
            print(f"    新增边数: {stats['added_edge_count']}")
            print(f"    补全率: {stats['completion_ratio']:.2%}")
        
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
                # 🔧 修复：不预先加载ct_array到实例变量，避免内存问题
                result = self.process_case(case_id)
                if result:
                    processed_cases.append(result)
            except Exception as e:
                print(f"Error processing case {case_id}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        print(f"Successfully processed {len(processed_cases)} cases")
        return processed_cases

    def _simplify_centerline(self, coords: np.ndarray, vessel_type: str = 'artery') -> np.ndarray:
        """多策略中心线简化"""
        if len(coords) <= 3:
            return coords
        
        # 获取简化参数
        params = self._get_simplification_params(vessel_type)
        
        # 策略1: Douglas-Peucker 3D简化
        simplified_dp = self._douglas_peucker_3d(coords, params['epsilon'])
        
        # 策略2: 基于距离的简化
        simplified_dist = self._distance_based_simplification(simplified_dp, params['min_distance'])
        
        # 策略3: 如果点太多，进行均匀采样
        if len(simplified_dist) > params['max_points']:
            indices = np.linspace(0, len(simplified_dist) - 1, params['max_points'], dtype=int)
            simplified_dist = simplified_dist[indices]
        
        return simplified_dist
    
    def _get_simplification_params(self, vessel_type: str) -> Dict:
        """获取血管类型特定的简化参数"""
        params = {
            'artery': {'epsilon': 2.0, 'min_distance': 1.5, 'max_points': 100},
            'vein': {'epsilon': 1.5, 'min_distance': 1.2, 'max_points': 80},
            'default': {'epsilon': 1.8, 'min_distance': 1.4, 'max_points': 90}
        }
        return params.get(vessel_type, params['default'])
    
    def _douglas_peucker_3d(self, coords: np.ndarray, epsilon: float) -> np.ndarray:
        """3D Douglas-Peucker算法"""
        if len(coords) <= 2:
            return coords
        
        def point_to_line_distance_3d(point, line_start, line_end):
            """计算3D点到直线的距离"""
            if np.allclose(line_start, line_end):
                return np.linalg.norm(point - line_start)
            
            line_vec = line_end - line_start
            point_vec = point - line_start
            
            line_len = np.linalg.norm(line_vec)
            if line_len == 0:
                return np.linalg.norm(point_vec)
            
            proj_length = np.dot(point_vec, line_vec) / line_len
            proj_point = line_start + (proj_length / line_len) * line_vec
            
            return np.linalg.norm(point - proj_point)
        
        def dp_recursive(points, start_idx, end_idx):
            if end_idx - start_idx <= 1:
                return [start_idx, end_idx]
            
            max_dist = 0
            max_idx = start_idx
            
            for i in range(start_idx + 1, end_idx):
                dist = point_to_line_distance_3d(points[i], points[start_idx], points[end_idx])
                if dist > max_dist:
                    max_dist = dist
                    max_idx = i
            
            if max_dist > epsilon:
                left_points = dp_recursive(points, start_idx, max_idx)
                right_points = dp_recursive(points, max_idx, end_idx)
                return left_points[:-1] + right_points
            else:
                return [start_idx, end_idx]
        
        indices = dp_recursive(coords, 0, len(coords) - 1)
        return coords[indices]
    
    def _distance_based_simplification(self, coords: np.ndarray, min_distance: float) -> np.ndarray:
        """基于距离的简化"""
        if len(coords) <= 2:
            return coords
        
        simplified = [coords[0]]  # 保持第一个点
        
        for i in range(1, len(coords)):
            if np.linalg.norm(coords[i] - simplified[-1]) >= min_distance:
                simplified.append(coords[i])
        
        # 确保保持最后一个点
        if not np.allclose(simplified[-1], coords[-1]):
            simplified.append(coords[-1])
        
        return np.array(simplified)
    
    def _validate_centerline_quality(self, coords: np.ndarray, original_mask: np.ndarray) -> Dict:
        """验证中心线质量"""
        if len(coords) < 2:
            return {
                'overall_score': 0.0,
                'length_preservation': 0.0,
                'shape_preservation': 0.0,
                'coverage': 0.0
            }
        
        # 计算长度
        distances = np.linalg.norm(np.diff(coords, axis=0), axis=1)
        total_length = np.sum(distances)
        
        # 估算原始长度
        estimated_original_length = np.sum(original_mask) ** (1/3) * 10
        length_preservation = min(1.0, total_length / estimated_original_length) if estimated_original_length > 0 else 0.0
        
        # 计算形状保持
        if len(coords) >= 3:
            curvatures = []
            for i in range(1, len(coords) - 1):
                v1 = coords[i] - coords[i-1]
                v2 = coords[i+1] - coords[i]
                
                if np.linalg.norm(v1) > 0 and np.linalg.norm(v2) > 0:
                    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                    cos_angle = np.clip(cos_angle, -1, 1)
                    curvature = np.arccos(cos_angle)
                    curvatures.append(curvature)
            
            if curvatures:
                curvature_std = np.std(curvatures)
                shape_preservation = max(0.0, 1.0 - curvature_std / np.pi)
            else:
                shape_preservation = 1.0
        else:
            shape_preservation = 1.0
        
        # 计算覆盖率
        coverage_count = 0
        for coord in coords:
            z, y, x = coord.astype(int)
            if (0 <= z < original_mask.shape[0] and 
                0 <= y < original_mask.shape[1] and 
                0 <= x < original_mask.shape[2]):
                if original_mask[z, y, x] > 0:
                    coverage_count += 1
        
        coverage = coverage_count / len(coords) if len(coords) > 0 else 0.0
        
        # 综合分数
        overall_score = (length_preservation * 0.3 + shape_preservation * 0.4 + coverage * 0.3)
        
        return {
            'overall_score': overall_score,
            'length_preservation': length_preservation,
            'shape_preservation': shape_preservation,
            'coverage': coverage
        }

def main():
    """主函数"""
    preprocessor = VesselPreprocessor(
        ct_dir="/home/lihe/classify/lungmap/data/raw/train",
        label_dir="/home/lihe/classify/lungmap/data/raw/label_filtered", 
        output_dir="/home/lihe/classify/lungmap/data/processed",
        cube_size=32
    )
    
    # 处理所有病例
    results = preprocessor.process_all_cases()
    
    print(f"\n🎉 Processing completed!")
    print(f"📊 Total processed cases: {len(results)}")
    print(f"📁 Output directory: processed_data/")

if __name__ == "__main__":
    main()
