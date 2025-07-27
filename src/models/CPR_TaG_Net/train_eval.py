
# === CPR-TaG-Net 训练与评估代码 ===
import os
import torch
import yaml
from torch.utils.data import DataLoader
from models.cpr_tagnet import CPRTaGNet
from data.dataset import VesselDataset
import torch.nn as nn
import pickle
from tqdm import tqdm
from data.utils import bresenham_3d

def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

# ========== 1. Train 函数 ==========
def train(model, dataloader, optimizer, criterion_label, criterion_seg=None, alpha=0.5):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for batch in tqdm(dataloader):
        # 由于batch_size=1，直接解包
        x_node, pos, edge_index, image_cubes, labels, seg_mask = batch
        
        # 去掉batch维度（因为batch_size=1）
        x_node = x_node.squeeze(0).cuda()
        pos = pos.squeeze(0).cuda() 
        edge_index = edge_index.squeeze(0).cuda()
        image_cubes = image_cubes.squeeze(0).cuda()
        labels = labels.squeeze(0).cuda()
        seg_mask = seg_mask.squeeze(0).cuda() if seg_mask is not None else None

        optimizer.zero_grad()
        outputs = model(x_node, pos, edge_index, image_cubes)

        if isinstance(outputs, tuple):
            logits, refined_seg = outputs
        else:
            logits = outputs
            refined_seg = None

        loss_label = criterion_label(logits, labels)
        if criterion_seg and refined_seg is not None:
            loss_seg = criterion_seg(refined_seg.squeeze(), seg_mask.float())
            loss = loss_label + alpha * loss_seg
        else:
            loss = loss_label

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x_node.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == labels).sum().item()
        total += labels.size(0)

    acc = correct / total
    avg_loss = total_loss / total
    return avg_loss, acc

# ========== 2. Validate 函数 ==========
def validate(model, dataloader, criterion_label):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in tqdm(dataloader):
            # 由于batch_size=1，直接解包
            x_node, pos, edge_index, image_cubes, labels, _ = batch
            
            # 去掉batch维度（因为batch_size=1）
            x_node = x_node.squeeze(0).cuda()
            pos = pos.squeeze(0).cuda() 
            edge_index = edge_index.squeeze(0).cuda()
            image_cubes = image_cubes.squeeze(0).cuda()
            labels = labels.squeeze(0).cuda()

            outputs = model(x_node, pos, edge_index, image_cubes)
            if isinstance(outputs, tuple):
                logits = outputs[0]
            else:
                logits = outputs

            loss = criterion_label(logits, labels)
            total_loss += loss.item() * x_node.size(0)

            pred = logits.argmax(dim=1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)
            all_preds.append(pred.cpu())
            all_labels.append(labels.cpu())

    acc = correct / total
    avg_loss = total_loss / total
    return avg_loss, acc, torch.cat(all_preds), torch.cat(all_labels)

# ========== 3. Graph Completion（推理阶段） ==========
def complete_graph(label_pred, node_pos, edge_index, label_rules):
    import networkx as nx
    G = nx.Graph()
    for i in range(edge_index.shape[1]):
        G.add_edge(edge_index[0, i].item(), edge_index[1, i].item())

    for lbl in torch.unique(label_pred):
        idx = torch.where(label_pred == lbl)[0].tolist()
        for i in idx:
            for j in idx:
                if i != j and not G.has_edge(i, j):
                    dist = torch.norm(node_pos[i] - node_pos[j])
                    if dist < 5.0:
                        G.add_edge(i, j)

    for (a, b) in label_rules:
        ai = torch.where(label_pred == a)[0].tolist()
        bi = torch.where(label_pred == b)[0].tolist()
        for i in ai:
            for j in bi:
                if not G.has_edge(i, j):
                    dist = torch.norm(node_pos[i] - node_pos[j])
                    if dist < 8.0:
                        G.add_edge(i, j)

    new_edges = torch.tensor(list(G.edges)).T.cuda()
    return new_edges

# ========== 4. 分割补全（基于 refined edge） ==========
def segmentation_completion(centerline_pos, refined_edge, volume_shape, radius=2):
    seg = torch.zeros(volume_shape, dtype=torch.uint8)
    for i, j in refined_edge.T:
        p1 = centerline_pos[i].cpu().numpy()
        p2 = centerline_pos[j].cpu().numpy()
        pts = bresenham_3d(p1, p2)
        for pt in pts:
            for dx in range(-radius, radius+1):
                for dy in range(-radius, radius+1):
                    for dz in range(-radius, radius+1):
                        x, y, z = pt[0]+dx, pt[1]+dy, pt[2]+dz
                        if 0 <= x < seg.shape[0] and 0 <= y < seg.shape[1] and 0 <= z < seg.shape[2]:
                            seg[x, y, z] = 1
    return seg

# ========== 5. 可视化与消融实验对照 ==========
def plot_confusion_matrix(y_true, y_pred, class_names):
    import seaborn as sns
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.show()


def main():
    # ==== 1. 加载配置 ====
    config = load_config('configs/train.yaml')
    os.makedirs(config['log']['save_dir'], exist_ok=True)

    # ==== 2. 加载数据 ====
    print("=== 加载数据 ===")
    
    # 检查数据文件是否存在
    if not os.path.exists(config['dataset']['train_data_path']):
        print(f"❌ 训练数据文件不存在: {config['dataset']['train_data_path']}")
        print("请先运行数据预处理脚本: python data/batch_preprocess.py")
        return
    
    if not os.path.exists(config['dataset']['val_data_path']):
        print(f"❌ 验证数据文件不存在: {config['dataset']['val_data_path']}")
        print("请先运行数据预处理脚本: python data/batch_preprocess.py")
        return
    
    with open(config['dataset']['train_data_path'], 'rb') as f:
        train_data = pickle.load(f)
    with open(config['dataset']['val_data_path'], 'rb') as f:
        val_data = pickle.load(f)

    print(f"✅ 成功加载数据:")
    print(f"   训练集: {len(train_data)} 个病例")
    print(f"   验证集: {len(val_data)} 个病例")
    
    # 打印数据统计信息
    total_train_nodes = sum(len(data['labels']) for data in train_data)
    total_val_nodes = sum(len(data['labels']) for data in val_data)
    
    print(f"   训练集节点总数: {total_train_nodes}")
    print(f"   验证集节点总数: {total_val_nodes}")
    
    # 检查标签分布
    all_labels = []
    for data in train_data:
        all_labels.extend(data['labels'].tolist())
    unique_labels = sorted(list(set(all_labels)))
    print(f"   标签类别: {unique_labels} (共{len(unique_labels)}类)")

    train_set = VesselDataset(train_data)
    val_set = VesselDataset(val_data)

    train_loader = DataLoader(train_set, batch_size=1, shuffle=True,
                              num_workers=config['dataset']['num_workers'])
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False)

    # ==== 3. 构建模型 ====
    print("\n=== 构建模型 ===")
    
    # 动态确定类别数
    num_classes = len(unique_labels) if len(unique_labels) <= 18 else 18
    print(f"模型类别数: {num_classes}")
    
    model = CPRTaGNet(
        num_classes=num_classes,
        node_feature_dim=config['model']['node_feature_dim'],
        image_channels=config['model']['image_condition_channels']
    ).cuda()
    
    print(f"✅ 模型构建完成")
    
    # 打印模型参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   总参数数: {total_params:,}")
    print(f"   可训练参数数: {trainable_params:,}")

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=config['training']['lr'],
                                 weight_decay=config['training']['weight_decay'])

    criterion_label = nn.CrossEntropyLoss()
    criterion_seg = nn.BCEWithLogitsLoss() if config['loss']['seg'] == "BCEWithLogits" else None

    # ==== 4. 训练循环 ====
    print(f"\n=== 开始训练 ===")
    print(f"训练配置:")
    print(f"   Epochs: {config['training']['epochs']}")
    print(f"   Learning Rate: {config['training']['lr']}")
    print(f"   Weight Decay: {config['training']['weight_decay']}")
    print(f"   Alpha (seg loss weight): {config['training']['alpha']}")
    
    best_val_acc = 0.0
    best_epoch = 0
    
    for epoch in range(config['training']['epochs']):
        print(f"\n--- Epoch {epoch + 1}/{config['training']['epochs']} ---")
        
        # 训练
        train_loss, train_acc = train(model, train_loader, optimizer, criterion_label, criterion_seg, config['training']['alpha'])
        
        # 验证
        val_loss, val_acc, val_preds, val_labels = validate(model, val_loader, criterion_label)

        print(f"[Train] Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
        print(f"[Val]   Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            torch.save(model.state_dict(), os.path.join(config['log']['save_dir'], "best_model.pth"))
            print(f"🎉 新的最佳验证准确率: {best_val_acc:.4f}")

        # 定期保存检查点
        if (epoch + 1) % config['log']['save_interval'] == 0:
            torch.save(model.state_dict(), os.path.join(config['log']['save_dir'], f"model_epoch{epoch + 1}.pth"))
            print(f"💾 保存检查点: epoch {epoch + 1}")
    
    print(f"\n=== 训练完成 ===")
    print(f"🏆 最佳验证准确率: {best_val_acc:.4f} (Epoch {best_epoch})")
    print(f"📁 模型保存路径: {config['log']['save_dir']}")
    
    # 最终评估
    print(f"\n=== 最终评估 ===")
    model.load_state_dict(torch.load(os.path.join(config['log']['save_dir'], "best_model.pth")))
    final_loss, final_acc, final_preds, final_labels = validate(model, val_loader, criterion_label)
    print(f"最终验证准确率: {final_acc:.4f}")
    
    # 打印混淆矩阵（如果类别不太多）
    if len(unique_labels) <= 10:
        try:
            from sklearn.metrics import classification_report
            report = classification_report(final_labels.numpy(), final_preds.numpy(), 
                                         target_names=[f"Class_{i}" for i in unique_labels])
            print("分类报告:")
            print(report)
        except ImportError:
            print("安装sklearn以查看详细分类报告: pip install scikit-learn")

if __name__ == "__main__":
    main()