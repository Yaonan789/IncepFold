import os
import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from metrics.metrics import insulation_pearson, pearson_correlation, mse, observed_vs_expected, distance_stratified_correlation
from model.DeepC import DeepC, ModelConfig
from model.corigami_models import ConvTransModel, ConvModel
from model.Akita import Akita
from model.Akita_multi import Akita_multi
from model.C_oigami_moe import Corigami_Moe
from model.inception import InceptionResNetDNA
from model.inception_10 import InceptionResNetDNA_10
from model.inception_14 import InceptionResNetDNA_14
from model.inception_decoder8 import InceptionResNetDNA_decoder8
from model.inception_decoder3 import InceptionResNetDNA_decoder3
from model.Orca import Orca
from preprocess.data_feature import HiCFeature, DNAFeature, GenomicFeature
from preprocess.get_dataset import GenomicDataset, collate_fn
from utils.plot_utils import MatrixPlot
from skimage.transform import resize
torch.manual_seed(42)
np.random.seed(42)


class Config:
    # 路径配置
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    use_aug = True
    # Corigami 的参数是 多模态参数数量
    epi=2
    genomic_features = True if epi > 0 else False
    input_dim=5+epi
    #model = Akita().to(device)
    #model = Corigami_Moe(2).to(device)
    model = InceptionResNetDNA(in_dim=input_dim).to(device)
    # model = Orca().to(device)
    #config = ModelConfig()
    #model = DeepC(config).to(device)
    # model = Akita().to(device)
    # model = Akita_multi().to(device)
    # model= ConvTransModel(2).to(device)
    species = 'Zea'
    windows = 2097152
    #windows = 2000000
    res = 10000
    output = 256
    # 路径配置
    # fasta_path = f'data/genome/{species}/genome.fa'
    # genomic_path = f'data/genomic_features/{species}/'
    # hic_dir = f"data/hic/{species}/"

    fasta_path = f'/home/user0/dpc_project/ChromatinPreditction/data/genome/{species}/Zm-B73-REFERENCE-GRAMENE-5.0.fa'
    
    # fasta_path = f'/home/user0/dpc_project/ChromatinPreditction/data/genome/{species}/M82_v1.fa'
    # fasta_path = f'/home/user0/dpc_project/ChromatinPreditction/data/genome/{species}/Sorghum-bicolor_BTx623.fa'
    genomic_path = f'/home/user0/dpc_project/ChromatinPreditction/data/genomic_features/{species}/'
    hic_dir = f"/home/user0/dpc_project/ChromatinPreditction/data/hic/{species}/"

    batch_size = 4

    # 模型保存路径
    model_path = f'output_oct/saved_models/{species}/{model.__class__.__name__}_{genomic_features}_1/best_model.pth'

    # 日志和结果保存
    log_dir = f"output_oct/logs/{species}/{model.__class__.__name__}_{genomic_features}_1/test/"
    os.makedirs(log_dir, exist_ok=True)
    results_file = os.path.join(log_dir, "test_results.txt")
    plot_dis_path = os.path.join(log_dir, "test_dis_plot.png")

config = Config()


def test_epoch(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []  # 存储所有预测值
    all_targets = []  # 存储所有真实值
    total_samples = 0

    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc="Test")
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


if __name__ == '__main__':
    # Corigami 的参数是 多模态参数数量
    genomic_features = True
    species = 'cotton'
    model = config.model
    #config = ModelConfig()
    #model = DeepC(config).to(device)
    #model = Akita().to(device)

    checkpoint = torch.load(config.model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print("Epoch:", checkpoint['epoch'])
    print("Val Loss:", checkpoint['val_loss'])
    print("Val Pearson:", checkpoint['val_pear'])
    print("val_insu_corr:", checkpoint['val_insu_corr'])

    windows = 2097152
    res = 10000
    output = 256

    test_dataset = GenomicDataset(
        fasta_path=config.fasta_path,
        hic_dir=config.hic_dir,
        genomic_path=config.genomic_path,
        mode='test',
        windows=config.windows,
        res=config.res,
        output=config.output,
        genomic_features=config.genomic_features
    )
    # 创建数据加载器
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )
    criterion = nn.MSELoss()  # 均方误差损失

    test_loss, test_insu_corr, test_mse, test_pear, test_oe, test_dis = test_epoch(
        model, test_loader, criterion, config.device
    )

    print(f"Best validation loss: {test_loss:.8f}")
    print(f"Best validation Mse: {test_mse:.8f}")
    print(f"Best validation Insu correlation: {test_insu_corr:.4f}")
    print(f"Best validation Pearson correlation: {test_pear:.4f}")
    print(f"Best validation Observed vs expected: {test_oe:.4f}")

    plt.figure(figsize=(12, 8))
    plt.plot(test_dis, marker='o', linestyle='-', color='b')

    # 添加标题和标签
    plt.title('baes_val_dises')
    plt.xlabel('The position from the diagonal')
    plt.ylabel('Pearson Correlation')
    # 显示网格
    plt.grid(True)
    # 显示图形
    plt.savefig(config.plot_dis_path)

    # 关闭数据集
    test_dataset.close()
    '''
    Moe:
    Best validation loss: 0.17325449
    Best validation Mse: 0.17325449
    Best validation Insu correlation: 0.9121
    Best validation Pearson correlation: 0.9239
    Best validation Observed vs expected: 0.6533
    '''