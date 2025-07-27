import os
import SimpleITK as sitk
import numpy as np
from typing import List, Dict, Set

def read_keep_labels(keep_file: str) -> List[str]:
    """读取需要保留的标签列表"""
    with open(keep_file, 'r', encoding='utf-8') as f:
        labels = [line.strip() for line in f if line.strip()]
    return labels

def get_label_mapping_from_seg_file(file_path: str) -> Dict[str, int]:
    """从.seg.nrrd文件中获取标签名称到值的映射"""
    try:
        reader = sitk.ImageFileReader()
        reader.SetFileName(file_path)
        reader.LoadPrivateTagsOn()
        reader.ReadImageInformation()
        
        label_mapping = {}
        
        # 尝试多种可能的标签键模式
        label_patterns = [
            'Segment{}_Name',
            'segment{}_name',
            'Segment{}_name', 
            'segment{}_Name',
            'Segmentation{}_Name',
            'label{}_name',
            'Label{}_Name'
        ]
        
        for pattern in label_patterns:
            i = 0
            found_any = False
            while i < 200:  # 扩大搜索范围
                label_key = pattern.format(i)
                value_key = pattern.replace('Name', 'LabelValue').replace('name', 'LabelValue').format(i)
                
                if reader.HasMetaDataKey(label_key):
                    label_name = reader.GetMetaData(label_key)
                    
                    # 获取对应的标签值
                    if reader.HasMetaDataKey(value_key):
                        label_value = int(reader.GetMetaData(value_key))
                    else:
                        label_value = i + 1  # 默认值
                    
                    label_mapping[label_name] = label_value
                    found_any = True
                    i += 1
                else:
                    if found_any:
                        i += 1
                        if i > 10:  # 连续10个索引都没找到就停止
                            break
                    else:
                        break
            
            if label_mapping:  # 如果找到了标签，就不用尝试其他模式了
                break
        
        return label_mapping
        
    except Exception as e:
        print(f"❌ Error reading {file_path}: {e}")
        return {}

def find_matching_labels(available_labels: List[str], keep_labels: List[str]) -> Dict[str, str]:
    """找到匹配的标签，处理命名变异"""
    matches = {}
    
    # 首先进行精确匹配
    for keep_label in keep_labels:
        if keep_label in available_labels:
            matches[keep_label] = keep_label
    
    # 对于未匹配的标签，尝试模糊匹配
    unmatched_keep = [label for label in keep_labels if label not in matches]
    unmatched_available = [label for label in available_labels if label not in matches.values()]
    
    # 模糊匹配规则
    for keep_label in unmatched_keep:
        best_match = None
        
        # 特殊匹配规则
        if keep_label == "L1+2":
            candidates = [label for label in unmatched_available if "L1+2" in label or ("L1" in label and "L2" in label)]
        elif keep_label == "L1+3":
            candidates = [label for label in unmatched_available if "L1+3" in label or ("L1" in label and "L3" in label)]
        elif keep_label == "R1+2":
            candidates = [label for label in unmatched_available if "R1+2" in label or ("R1" in label and "R2" in label)]
        elif keep_label == "R1+3":
            candidates = [label for label in unmatched_available if "R1+3" in label or ("R1" in label and "R3" in label)]
        elif keep_label == "Linternal":
            candidates = [label for label in unmatched_available if "Linter" in label or "Linternal" in label or "Lintrenal" in label]
        elif keep_label == "Rinternal":
            candidates = [label for label in unmatched_available if "Rinter" in label or "Rinternal" in label or "Rintrenal" in label]
        elif keep_label == "Lupper":
            candidates = [label for label in unmatched_available if "Lupper" in label or "L-upper" in label]
        elif keep_label == "Rupper":
            candidates = [label for label in unmatched_available if "Rupper" in label or "R-upper" in label or "URPA" in label]
        elif keep_label == "Ldown":
            candidates = [label for label in unmatched_available if "Ldown" in label or "L-down" in label]
        elif keep_label == "RDown":
            candidates = [label for label in unmatched_available if "Rdown" in label or "R-down" in label or "RDown" in label]
        elif keep_label == "Lmedium":
            candidates = [label for label in unmatched_available if "Lmedium" in label or "Lmedian" in label or "L-medium" in label]
        elif keep_label == "Rmedium":
            candidates = [label for label in unmatched_available if "Rmedium" in label or "Rmedian" in label or "R-medium" in label]
        else:
            # 通用匹配：寻找包含目标标签的候选项
            candidates = [label for label in unmatched_available if keep_label.lower() in label.lower()]
        
        if candidates:
            # 选择最相似的候选项（最短的通常是最好的匹配）
            best_match = min(candidates, key=len)
            matches[keep_label] = best_match
            unmatched_available.remove(best_match)
    
    return matches

def create_filtered_segmentation(input_file: str, output_file: str, keep_labels: List[str]) -> bool:
    """创建只包含指定标签的分割文件"""
    try:
        print(f"正在处理: {os.path.basename(input_file)}")
        
        # 读取原始图像
        image = sitk.ReadImage(input_file)
        image_array = sitk.GetArrayFromImage(image)
        
        # 获取标签映射
        label_mapping = get_label_mapping_from_seg_file(input_file)
        if not label_mapping:
            print(f"  ❌ 无法获取标签映射")
            return False
        
        available_labels = list(label_mapping.keys())
        print(f"  📋 原始标签数量: {len(available_labels)}")
        
        # 找到匹配的标签
        matches = find_matching_labels(available_labels, keep_labels)
        print(f"  ✅ 找到匹配标签: {len(matches)}")
        
        # 显示匹配结果
        for keep_label, matched_label in matches.items():
            if keep_label != matched_label:
                print(f"    {keep_label} -> {matched_label}")
            else:
                print(f"    {keep_label} ✓")
        
        # 显示未匹配的标签
        unmatched = [label for label in keep_labels if label not in matches]
        if unmatched:
            print(f"  ⚠️  未匹配的标签: {unmatched}")
        
        # 创建新的标签值映射 (保持原始标签值)
        keep_label_values = {}
        for keep_label, matched_label in matches.items():
            original_value = label_mapping[matched_label]
            keep_label_values[original_value] = original_value
        
        # 创建新的图像数组，只保留指定的标签
        new_image_array = np.zeros_like(image_array)
        
        for old_value, new_value in keep_label_values.items():
            mask = (image_array == old_value)
            new_image_array[mask] = new_value
            pixel_count = np.sum(mask)
            print(f"    标签值 {old_value}: {pixel_count} 像素")
        
        # 创建新的图像
        new_image = sitk.GetImageFromArray(new_image_array)
        new_image.CopyInformation(image)
        
        # 复制原始元数据并更新标签信息
        reader = sitk.ImageFileReader()
        reader.SetFileName(input_file)
        reader.LoadPrivateTagsOn()
        reader.ReadImageInformation()
        
        # 复制基本元数据
        for key in reader.GetMetaDataKeys():
            if not key.startswith('Segment'):
                try:
                    new_image.SetMetaData(key, reader.GetMetaData(key))
                except:
                    pass
        
        # 添加保留的标签的元数据
        segment_index = 0
        for keep_label, matched_label in matches.items():
            original_value = label_mapping[matched_label]
            
            # 寻找原始标签的元数据
            original_segment_index = None
            for i in range(200):
                name_key = f'Segment{i}_Name'
                value_key = f'Segment{i}_LabelValue'
                if (reader.HasMetaDataKey(name_key) and 
                    reader.HasMetaDataKey(value_key) and
                    reader.GetMetaData(name_key) == matched_label and
                    int(reader.GetMetaData(value_key)) == original_value):
                    original_segment_index = i
                    break
            
            if original_segment_index is not None:
                # 复制原始标签的所有元数据，但使用新的索引
                for meta_key in reader.GetMetaDataKeys():
                    if meta_key.startswith(f'Segment{original_segment_index}_'):
                        new_key = meta_key.replace(f'Segment{original_segment_index}_', f'Segment{segment_index}_')
                        try:
                            # 保持原始标签名称
                            if new_key.endswith('_Name'):
                                new_image.SetMetaData(new_key, keep_label)  # 使用保留的标签名称
                            else:
                                new_image.SetMetaData(new_key, reader.GetMetaData(meta_key))
                        except:
                            pass
                
                segment_index += 1
        
        # 保存新图像
        writer = sitk.ImageFileWriter()
        writer.SetFileName(output_file)
        writer.UseCompressionOn()
        writer.Execute(new_image)
        
        print(f"  ✅ 已保存到: {os.path.basename(output_file)}")
        return True
        
    except Exception as e:
        print(f"  ❌ 处理失败: {e}")
        return False

def process_all_seg_files(input_dir: str, output_dir: str, keep_file: str):
    """处理所有的.seg.nrrd文件"""
    # 读取需要保留的标签
    keep_labels = read_keep_labels(keep_file)
    print(f"🎯 需要保留的标签 ({len(keep_labels)}个):")
    for i, label in enumerate(keep_labels, 1):
        print(f"  {i:2d}. {label}")
    print()
    
    # 创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"📁 创建输出目录: {output_dir}")
    
    # 获取所有.seg.nrrd文件
    seg_files = [f for f in os.listdir(input_dir) if f.endswith('.seg.nrrd')]
    print(f"📄 找到 {len(seg_files)} 个分割文件\n")
    
    success_count = 0
    total_count = len(seg_files)
    
    for i, filename in enumerate(seg_files, 1):
        print(f"[{i:2d}/{total_count}] ", end="")
        input_file = os.path.join(input_dir, filename)
        output_file = os.path.join(output_dir, filename)
        
        if create_filtered_segmentation(input_file, output_file, keep_labels):
            success_count += 1
        print()
    
    print(f"🎉 处理完成！成功处理 {success_count}/{total_count} 个文件")
    print(f"📁 输出目录: {output_dir}")

if __name__ == "__main__":
    input_directory = "label"  # 输入目录
    output_directory = "label_filtered"  # 输出目录
    keep_file = "keep.txt"  # 保留标签文件
    
    process_all_seg_files(input_directory, output_directory, keep_file)
