import os
import SimpleITK as sitk

def get_label_info_from_seg_file(file_path):
    try:
        # 读取 .seg.nrrd 文件
        reader = sitk.ImageFileReader()
        reader.SetFileName(file_path)
        reader.LoadPrivateTagsOn()  # 关键！读取元数据中的标签信息
        reader.ReadImageInformation()
        
        # 打印所有可用的元数据键以便调试
        print(f"🔍 调试信息 - {os.path.basename(file_path)}:")
        all_keys = reader.GetMetaDataKeys()
        print(f"   总元数据键数量: {len(all_keys)}")
        
        # 显示前20个键作为参考
        if all_keys:
            print("   前20个元数据键:")
            for key in all_keys[:20]:
                try:
                    value = reader.GetMetaData(key)
                    print(f"     {key}: {value}")
                except:
                    print(f"     {key}: <无法读取>")
        
        # 尝试多种可能的标签键模式
        label_names = []
        label_patterns = [
            'segment{}_name',
            'Segment{}_Name', 
            'Segmentation{}_Name',
            'label{}_name',
            'Label{}_Name',
            'segment_{}_name',
            'Segment_{}_Name'
        ]
        
        # 尝试不同的键模式
        for pattern in label_patterns:
            i = 0
            found_any = False
            while i < 100:  # 限制搜索范围，避免无限循环
                label_key = pattern.format(i)
                if reader.HasMetaDataKey(label_key):
                    label_names.append(reader.GetMetaData(label_key))
                    found_any = True
                    i += 1
                else:
                    if found_any:  # 如果找到过标签但当前索引没有，继续尝试下一个
                        i += 1
                        if i > 10:  # 连续10个索引都没找到就停止
                            break
                    else:
                        break
            
            if label_names:  # 如果找到了标签，就不用尝试其他模式了
                print(f"   ✅ 使用模式 '{pattern}' 找到标签")
                break
        
        # 如果没找到标签，尝试读取图像并分析唯一值
        if not label_names:
            print("   🔄 未找到标签元数据，尝试分析图像像素值...")
            image = sitk.ReadImage(file_path)
            
            # 获取图像的统计信息
            stats_filter = sitk.LabelStatisticsImageFilter()
            stats_filter.Execute(image, image)
            
            unique_labels = stats_filter.GetLabels()
            print(f"   图像中的唯一标签值: {list(unique_labels)}")
            
            # 如果有多个标签值，创建通用标签名
            if len(unique_labels) > 1:
                label_names = [f"Label_{label}" for label in unique_labels if label != 0]  # 排除背景(0)
                print(f"   📝 创建通用标签名: {label_names}")

        return label_names
        
    except Exception as e:
        print(f"❌ Error reading {file_path}: {e}")
        return []

def scan_directory_for_seg_files(path):
    seg_files = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.seg.nrrd')]
    
    print(f"🔍 在目录中找到 {len(seg_files)} 个 .seg.nrrd 文件\n")
    
    for file in seg_files:
        labels = get_label_info_from_seg_file(file)
        print(f"📄 {os.path.basename(file)}:")
        print(f"   标签数量: {len(labels)}")
        print(f"   标签名称: {labels}")
        print("-" * 80)
        print()

# 扫描当前目录下的 label 文件夹
path = "label"
scan_directory_for_seg_files(path)