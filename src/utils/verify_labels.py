import os
import SimpleITK as sitk

def verify_filtered_labels(directory: str, keep_file: str):
    """验证过滤后的标签文件"""
    
    # 读取期望的标签
    with open(keep_file, 'r', encoding='utf-8') as f:
        expected_labels = [line.strip() for line in f if line.strip()]
    
    print(f"🎯 期望保留的标签 ({len(expected_labels)}个):")
    for i, label in enumerate(expected_labels, 1):
        print(f"  {i:2d}. {label}")
    print()
    
    # 获取所有处理后的文件
    seg_files = [f for f in os.listdir(directory) if f.endswith('.seg.nrrd')]
    print(f"📄 验证 {len(seg_files)} 个过滤后的文件\n")
    
    summary = {}
    
    for i, filename in enumerate(seg_files, 1):
        print(f"[{i:2d}/{len(seg_files)}] {filename}")
        file_path = os.path.join(directory, filename)
        
        try:
            # 读取标签信息
            reader = sitk.ImageFileReader()
            reader.SetFileName(file_path)
            reader.LoadPrivateTagsOn()
            reader.ReadImageInformation()
            
            # 获取标签名称
            labels = []
            for j in range(100):
                label_key = f'Segment{j}_Name'
                if reader.HasMetaDataKey(label_key):
                    labels.append(reader.GetMetaData(label_key))
            
            print(f"  📋 保留的标签 ({len(labels)}个): {', '.join(labels)}")
            
            # 统计标签
            for label in labels:
                if label in summary:
                    summary[label] += 1
                else:
                    summary[label] = 1
                    
            # 读取图像检查像素值
            image = sitk.ReadImage(file_path)
            image_array = sitk.GetArrayFromImage(image)
            unique_values = list(set(image_array.flatten()))
            unique_values.sort()
            non_zero_values = [v for v in unique_values if v != 0]
            
            print(f"  🏷️  标签值: {non_zero_values}")
            print()
            
        except Exception as e:
            print(f"  ❌ 读取失败: {e}")
            print()
    
    print("📊 标签统计汇总:")
    print(f"{'标签名称':<15} {'出现次数':<8} {'覆盖率'}")
    print("-" * 35)
    
    for label in expected_labels:
        count = summary.get(label, 0)
        coverage = f"{count}/{len(seg_files)}"
        print(f"{label:<15} {count:<8} {coverage}")
    
    print(f"\n总文件数: {len(seg_files)}")
    print(f"期望标签数: {len(expected_labels)}")
    print(f"实际找到的不同标签数: {len(summary)}")

if __name__ == "__main__":
    verify_filtered_labels("label_filtered", "keep.txt")
