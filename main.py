import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr
import numpy as np
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
###############################################
from preprocess.get_dataset import GenomicDataset, collate_fn
from model.corigami_models import ConvTransModel, ConvModel
from model.C_oigami_moe import Corigami_Moe
from model.Moe_Inception import MoeInceptionResNetDNA
from model.Moe_Inception1 import MoeInception1
from model.inception import InceptionResNetDNA
from model.inception_10 import InceptionResNetDNA_10
from model.inception_14 import InceptionResNetDNA_14
from model.inception_decoder8 import InceptionResNetDNA_decoder8
from model.inception_decoder3 import InceptionResNetDNA_decoder3
from model.Akita import Akita
from model.Orca import Orca
from model.DeepC import ModelConfig, DeepC
from metrics.metrics import insulation_pearson,mse,pearson_correlation,observed_vs_expected,distance_stratified_correlation
# 设置随机种子以确保可重复性
torch.manual_seed(42)
np.random.seed(42)

# 配置
class Config:
    # 路径配置
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    use_aug = True
    # 模型的参数是 多模态参数数量
    epi = 2
    genomic_features = True if epi > 0 else False
    print(genomic_features)
    # model = ConvTransModel(2).to(device)
    #model = MoeInception1(epi).to(device)  # 1
    input_dim=5+epi
    model = InceptionResNetDNA(in_dim=input_dim).to(device)
    print(f'~~~~~~~~~~~~~~~~~~input_dim:{input_dim}~~~~~~~~~~~~~~~')
    # model = Orca().to(device)
    #config = ModelConfig()
    #model = DeepC(config).to(device)
    # model = Akita().to(device)
    species = 'Zea'
    windows = 2097152
    # windows = 2000000
    res = 10000
    output = 256
    # 路径配置
    fasta_path = f'/home/user0/dpc_project/ChromatinPreditction/data/genome/{species}/Zm-B73-REFERENCE-GRAMENE-5.0.fa'
    # fasta_path = f'/home/user0/dpc_project/ChromatinPreditction/data/genome/{species}/M82_v1.fa'

    # fasta_path = f'/home/user0/dpc_project/ChromatinPreditction/data/genome/{species}/Sorghum-bicolor_BTx623.fa'
    genomic_path = f'/home/user0/dpc_project/ChromatinPreditction/data/genomic_features/{species}/'
    hic_dir = f"/home/user0/dpc_project/ChromatinPreditction/data/hic/{species}/"

    # fasta_path = f'data/genome/{species}/genome.fa'
    # genomic_path = f'data/genomic_features/{species}/'
    # hic_dir = f"data/hic/{species}/"

    # 训练参数
    batch_size = 4
    learning_rate = 2e-4
    weight_decay = 1e-5
    epochs = 100
    patience = 20  # 早停的耐心值

    # 模型保存路径
    model_dir = f"output_oct/saved_models/{species}/{model.__class__.__name__}_{genomic_features}_1/"
    os.makedirs(model_dir, exist_ok=True)
    best_model_path = os.path.join(model_dir, "best_model.pth")

    # 日志和结果保存
    log_dir = f"output_oct/logs/{species}/{model.__class__.__name__}_{genomic_features}_1/"
    os.makedirs(log_dir, exist_ok=True)
    results_file = os.path.join(log_dir, "training_results.txt")
    plot_file = os.path.join(log_dir, "training_plot.png")
    plot_dis_path = os.path.join(log_dir, "val_dis_plot.png")

config = Config()

def benchmark_model(model, config, criterion, optimizer):
    """
    计算并打印给定模型的关键性能和效率指标。
    """
    print("\n" + "="*60)
    print("PERFORMANCE AND EFFICIENCY BENCHMARK".center(60))
    print("="*60)

    try:
        from fvcore.nn import FlopCountAnalysis
    except ImportError:
        print("请安装 fvcore 以进行 FLOPs 分析: pip install fvcore")
        FlopCountAnalysis = None

    device = config.device
    model.to(device)

    # 1. 模型名称
    model_name = model.__class__.__name__
    print(f"{'Model':<35}: {model_name}")

    # 2. 模型参数量 (M)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    m_params = trainable_params / 1e6
    print(f"{'Parameters (M)':<35}: {m_params:.2f}")

    # ############################ 最终修正点 ##############################
    # 修正虚拟输入的形状为模型期望的 (N, L, C) 格式
    # 即 (batch_size, sequence_length, channels)
    # ##################################################################
    correct_input_shape_flops = (1, config.windows, config.input_dim)
    correct_input_shape_runtime = (config.batch_size, config.windows, config.input_dim)

    # 3. 理论计算密度 (GFLOPs/M)
    if FlopCountAnalysis:
        dummy_input_flops = torch.randn(correct_input_shape_flops).to(device)
        try:
            # 某些模型可能需要在内部转换维度，这里直接传递
            flops_analyzer = FlopCountAnalysis(model, dummy_input_flops)
            gflops = flops_analyzer.total() / 1e9
            gflops_per_m = gflops / m_params if m_params > 0 else 0
            print(f"{'Inference (GFLOPs/M)':<35}: {gflops_per_m:.2f}")
        except Exception as e:
            print(f"{'Inference (GFLOPs/M)':<35}: 计算失败 ({e})")
    
    # 4. Batch Size
    batchsize = config.batch_size
    print(f"{'Batch Size':<35}: {batchsize}")

    # 使用正确的形状创建用于运行时测试的虚拟输入
    dummy_input_runtime = torch.randn(correct_input_shape_runtime).to(device)

    # 5. 训练时的峰值GPU显存 (GB)
    model.train()
    optimizer.zero_grad()
    torch.cuda.reset_peak_memory_stats(device)
    
    try:
        outputs = model(dummy_input_runtime)
        # 假设输出也是 (N, L_out, C_out) 或类似格式，创建一个匹配的dummy_target
        # 注意：这里的dummy_target形状可能需要根据模型实际输出来调整
        dummy_target = torch.randn_like(outputs)
        loss = criterion(outputs, dummy_target)
        loss.backward()
        optimizer.step()
        
        peak_mem_train_gb = torch.cuda.max_memory_allocated(device) / (1024**3)
        print(f"{'Peak GPU Memory (GB) Train':<35}: {peak_mem_train_gb:.2f}")

    except Exception as e:
        print(f"{'Peak GPU Memory (GB) Train':<35}: 计算失败 ({e})")
    
    finally:
        optimizer.zero_grad()
        torch.cuda.empty_cache()

    print("-" * 60)
    print("基准测试完成，即将开始正式训练...".center(60))
    print("="*60 + "\n")

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    all_preds = []  # 存储所有预测值
    all_targets = []  # 存储所有真实值
    total_samples = 0
    progress_bar = tqdm(dataloader, desc="Training")
    for batch_idx, (dna, hic) in enumerate(progress_bar):
        # 移动到设备
        dna = dna.to(device)
        hic_target = hic.to(device)

        # 前向传播
        optimizer.zero_grad()
        outputs = model(dna)

        # 计算损失
        loss = criterion(outputs, hic_target)

        # 反向传播和优化
        loss.backward()
        optimizer.step()
        # 更新统计信息
        batch_size = dna.size(0)
        batch_loss = loss.item()
        running_loss += batch_loss * batch_size
        total_samples += batch_size

        # 收集预测和真实值
        all_preds.extend(outputs.detach().cpu().numpy())
        all_targets.extend(hic_target.detach().cpu().numpy())

        # 计算当前累计平均值
        avg_loss = running_loss / total_samples

        # 更新进度条 - 显示当前批次的损失
        progress_bar.set_postfix({
            'batch_loss': f"{batch_loss:.8f}",  # 当前批次的损失
            'avg_loss': f"{avg_loss:.8f}"  # 累计平均损失
        })

    if total_samples == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    all_preds_array = np.array(all_preds)  # 形状为 [N, 256, 256]
    all_targets_array = np.array(all_targets)  # 形状为 [N, 256, 256]
    avg_insu = np.nanmean(insulation_pearson(all_preds_array, all_targets_array))  # 使用 nanmean 忽略 NaN 值
    avg_mse = np.nanmean(mse(all_preds_array, all_targets_array))
    avg_pear = np.nanmean(pearson_correlation(all_preds_array, all_targets_array))
    avg_oe = np.nanmean(observed_vs_expected(all_preds_array, all_targets_array))
    epoch_loss = running_loss / total_samples
    return epoch_loss, avg_insu, avg_mse, avg_pear, avg_oe



def validate_epoch(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []  # 存储所有预测值
    all_targets = []  # 存储所有真实值
    total_samples = 0

    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc="Validation")
        for batch_idx, (dna, hic) in enumerate(progress_bar):
            # 移动到设备
            dna = dna.to(device)
            hic_target = hic.to(device)

            # 前向传播
            outputs = model(dna)

            # 计算损失
            loss = criterion(outputs, hic_target)

            # 更新统计信息
            batch_size = dna.size(0)
            batch_loss = loss.item()
            running_loss += batch_loss * batch_size
            total_samples += batch_size

            # 收集预测和真实值
            all_preds.extend(outputs.detach().cpu().numpy())
            all_targets.extend(hic_target.detach().cpu().numpy())

            # 计算当前累计平均值
            avg_loss = running_loss / total_samples

            # 更新进度条 - 显示当前批次的损失
            progress_bar.set_postfix({
                'batch_loss': f"{batch_loss:.8f}",
                'avg_loss': f"{avg_loss:.8f}"
            })

    if total_samples == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0


    # 计算整个epoch的平均绝缘相关系数
    all_preds_array = np.array(all_preds)  # 形状为 [N, 256, 256]
    all_targets_array = np.array(all_targets)  # 形状为 [N, 256, 256]
    avg_insu = np.nanmean(insulation_pearson(all_preds_array, all_targets_array))  # 使用 nanmean 忽略 NaN 值
    avg_mse = np.nanmean(mse(all_preds_array, all_targets_array))
    avg_pear = np.nanmean(pearson_correlation(all_preds_array, all_targets_array))
    avg_oe = np.nanmean(observed_vs_expected(all_preds_array, all_targets_array))
    avg_dis = np.nanmean(distance_stratified_correlation(all_preds_array, all_targets_array),axis=0)
    epoch_loss = running_loss / total_samples

    return epoch_loss, avg_insu, avg_mse, avg_pear, avg_oe, avg_dis


# 主训练流程
def main():
    # 初始化完整数据集
    train_dataset = GenomicDataset(
        fasta_path=config.fasta_path,
        hic_dir=config.hic_dir,
        genomic_path=config.genomic_path,
        mode='train',
        windows=config.windows,
        res=config.res,
        output=config.output,
        genomic_features=config.genomic_features,
        use_aug=config.use_aug
    )

    val_dataset = GenomicDataset(
        fasta_path=config.fasta_path,
        hic_dir=config.hic_dir,
        genomic_path=config.genomic_path,
        mode='valid',
        windows=config.windows,
        res=config.res,
        output=config.output,
        genomic_features=config.genomic_features
    )
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )

    # 初始化模型、损失函数和优化器
    model = config.model
    print(f'config.device:{config.device}')
    print(f"Initial Model :{config.model.__class__.__name__} !!!")
    criterion = nn.MSELoss()  # 均方误差损失
    optimizer = optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )

    benchmark_model(model, config, criterion, optimizer)

    # 学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )

    # 训练记录 train_loss, train_insu_corr,train_mse,train_pear,train_oe,train_dis
    train_losses = []
    train_insu_corrs = []
    train_mses = []
    train_pears = []
    train_oes = []
    val_losses = []
    val_insu_corrs = []
    val_mses = []
    val_pears = []
    val_oes = []

    best_val_loss = float('inf')
    best_val_corr = 0.0
    best_val_mse = 0.0
    best_val_pear = 0.0
    best_val_os = 0.0
    baes_val_dises = []
    epochs_without_improvement = 0

    # 打开结果文件
    with open(config.results_file, 'w') as f:
        f.write("Epoch\tTrain Loss\tVal Loss\tTrain insu Corr\tVal insu Corr\tTrain Mse\tVal Mse\tTrain Pearson\tVal Pearson\t"
                "Train OE\tVal OE\tTime\n")

    # 训练循环
    start_time = time.time()
    print(f'################ Start train!!!! ######################')
    for epoch in range(config.epochs):
        epoch_start = time.time()

        print(f"\nEpoch {epoch + 1}/{config.epochs}")
        print("-" * 50)

        # 训练一个epoch
        train_loss, train_insu_corr,train_mse,train_pear,train_oe = train_epoch(
            model, train_loader, criterion, optimizer, config.device
        )

        # 验证一个epoch
        val_loss, val_insu_corr,val_mse,val_pear,val_oe,val_dis = validate_epoch(
            model, val_loader, criterion, config.device
        )

        # 记录结果
        train_losses.append(train_loss)
        train_insu_corrs.append(train_insu_corr)
        train_mses.append(train_mse)
        train_pears.append(train_pear)
        train_oes.append(train_oe)

        val_losses.append(val_loss)
        val_insu_corrs.append(val_insu_corr)
        val_mses.append(val_mse)
        val_pears.append(val_pear)
        val_oes.append(val_oe)


        # 计算epoch耗时
        epoch_time = time.time() - epoch_start

        # 打印结果
        print(f"\nEpoch {epoch + 1} Summary:")
        print(f"  Train Loss: {train_loss:.8f}, Val Loss: {val_loss:.8f}")
        print(f"  Train Mse: {train_mse:.8f}, Val Mse: {val_mse:.8f}")
        print(f"  Train Insu corrs: {train_insu_corr:.4f}, Val Insu corrs: {val_insu_corr:.4f}")
        print(f"  Train Pearson: {train_pear:.4f}, Val Pearson: {val_pear:.4f}")
        print(f"  Train Observed vs expected: {train_oe:.4f}, Val Observed vs expected: {val_oe:.4f}")
        print(f"  Time: {epoch_time:.2f} seconds")

        # 保存到文件
        with open(config.results_file, 'a') as f:
            f.write(
                f"{epoch + 1}\t{train_loss:.8f}\t{val_loss:.8f}\t{train_insu_corr:.4f}\t{val_insu_corr:.4f}\t"
                f"{train_mse:.8f}\t{val_mse:.8f}\t{train_pear:.4f}\t{val_pear:.4f}\t"
                f"{train_oe:.4f}\t{val_oe:.4f}\t{epoch_time:.2f}\n")

        # 更新学习率
        scheduler.step(val_loss)

        # 检查是否是最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_corr = val_insu_corr
            best_val_mse = val_mse
            best_val_pear = val_pear
            best_val_os = val_oe
            baes_val_dises = val_dis
            epochs_without_improvement = 0

            # 保存最佳模型
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'train_insu_corr': train_insu_corr,
                'val_insu_corr': val_insu_corr,
                'train_mse': train_mse,
                'val_mse': val_mse,
                'train_pear': train_pear,
                'val_pear': val_pear,
                'train_oe': train_oe,
                'val_oe': val_oe,
            }, config.best_model_path)
            print(f"Saved best model with val loss {val_loss:.8f}")
        else:
            epochs_without_improvement += 1
            print(f"No improvement for {epochs_without_improvement}/{config.patience} epochs")

        # 检查早停条件
        if epochs_without_improvement >= config.patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    # 训练结束
    total_time = time.time() - start_time
    print(f"\nTraining complete in {total_time:.2f} seconds")
    print(f"Best validation loss: {best_val_loss:.8f}")
    print(f"Best validation Mse: {best_val_mse:.8f}")
    print(f"Best validation Insu correlation: {best_val_corr:.4f}")
    print(f"Best validation Pearson correlation: {best_val_pear:.4f}")
    print(f"Best validation Observed vs expected: {best_val_os:.4f}")

    # 绘制训练曲线
    plt.figure(figsize=(12, 8))

    # 损失曲线
    plt.subplot(2, 1, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)

    # Pearson相关系数曲线
    plt.subplot(2, 1, 2)
    plt.plot(train_insu_corrs, label='Train Pearson')  # 平均值的列表
    plt.plot(val_insu_corrs, label='Validation Pearson')
    plt.xlabel('Epochs')
    plt.ylabel('Pearson Correlation')
    plt.title('Training and Validation Pearson Correlation')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(config.plot_file)
    plt.close()

    # 距离相关系数图
    plt.figure(figsize=(12, 8))
    plt.plot(baes_val_dises, marker='o', linestyle='-', color='b')

    # 添加标题和标签
    plt.title('baes_val_dises')
    plt.xlabel('The position from the diagonal')
    plt.ylabel('Pearson Correlation')
    # 显示网格
    plt.grid(True)
    # 显示图形
    plt.savefig(config.plot_dis_path)

    # 关闭数据集
    train_dataset.close()
    val_dataset.close()



if __name__ == "__main__":
    main()