import os
import numpy as np
import torch
from metrics.metrics import insulation_pearson, pearson_correlation,mse, observed_vs_expected
from model.DeepC import DeepC, ModelConfig
from model.corigami_models import ConvTransModel, ConvModel
from model.Akita import Akita
from model.C_oigami_moe import Corigami_Moe
from model.inception import InceptionResNetDNA
from model.inception_ctcf import InceptionResNetDNA_ctcf
from model.Orca import Orca
from preprocess.data_feature import HiCFeature, DNAFeature, GenomicFeature
from utils.plot_utils import MatrixPlot
from skimage.transform import resize
torch.manual_seed(42)
np.random.seed(42)


if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # Corigami 的参数是 多模态参数数量
    genomic_features = False
    species = 'Zea'
    #model = Corigami_Moe(2).to(device)    #用
    #config = ModelConfig()
    #model = DeepC(config).to(device
    # model = InceptionResNetDNA().to(device)
    # model= InceptionResNetDNA_ctcf().to(device)
    #model=ConvTransModel(2).to(device)
    #model=Akita().to(device)
    model=Orca().to(device)
    model_name = model.__class__.__name__
    checkpoint = torch.load(f'output_oct/saved_models/{species}/{model_name}_{genomic_features}/best_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print("Epoch:", checkpoint['epoch'])
    print("Val Loss:", checkpoint['val_loss'])
    print("Val Pearson:", checkpoint['val_pear'])
    print("val_insu_corr:", checkpoint['val_insu_corr'])
    #print("o/e corr:", checkpoint['val_'])

    chrom = '8'
    windows = 2097152
    res = 10000
    output = 256
    '''s = 29489440
    e = 26497632
    c = (s+e)//2
    seq_start = c - windows//2'''
    start_result = {}
    insu_result = {}
    pear_resulr = {}
    for seq_start in range(0, 100000000, 2097152):
        #seq_start = 20444960
        print(f'seq_start:{seq_start}')

        seq_end = seq_start + windows
        # 路径配置
        # fasta_path = f'data/genome/{species}/l128.chr.fa'
        # genomic_path = f'data/genomic_features/{species}/'
        # hic_dir = f"data/hic/{species}/"
        fasta_path = f'/home/user0/dpc_project/ChromatinPreditction/data/genome/{species}/Zm-B73-REFERENCE-GRAMENE-5.0.fa'
        genomic_path = f'/home/user0/dpc_project/ChromatinPreditction/data/genomic_features/{species}/'
        hic_dir = f"/home/user0/dpc_project/ChromatinPreditction/data/hic/{species}/"
        path = hic_dir + f'Zm_B73_Chr{chrom}.npz'
        output_path = f'output_oct/hic_fig/{species}/{model_name}_{genomic_features}/'
        os.makedirs(output_path, exist_ok=True)

        dna_feature = DNAFeature(path=fasta_path)
        atac_feater = GenomicFeature(path=genomic_path + 'atac.bw', norm='log')
        ctcf_feater = GenomicFeature(path=genomic_path + 'h3k4me3.bw', norm='log')

        dna = dna_feature.get(chrom, seq_start, seq_end)
        deature_tensor = torch.tensor(dna, dtype=torch.float32)  # [self.windows,5]
        input_tensor = deature_tensor.unsqueeze(0).to(device)
        if genomic_features:
            atac = atac_feater.get(chrom, seq_start, seq_end).reshape(-1, 1)
            ctcf = ctcf_feater.get(chrom, seq_start, seq_end).reshape(-1, 1)  # [self.windows,1]
            print(f'ctcf max:{np.max(ctcf)} ctcf min{np.min(ctcf)}')
            print(f'atac max::{np.max(atac)} atac min{np.min(atac)}')

            combined_features = np.concatenate((dna, atac, ctcf), axis=1)
            #combined_features = np.concatenate((dna, ctcf), axis=1)

            input_tensor = torch.tensor(combined_features, dtype=torch.float32).unsqueeze(0).to(device)

        print(f"input_tensor.shape:{input_tensor.shape}")

        hic_feature = HiCFeature(path=path)
        targets = hic_feature.get(seq_start, windows, res)
        targets = resize(targets, (256, 256), anti_aliasing=True)
        targets = np.log(targets + 1)

        #print(f'input_tensor:max{max(input_tensor)},min{min(input_tensor)}')

        pre = model(input_tensor)
        pre = pre.squeeze(0).detach().cpu().numpy()

        a_insu = insulation_pearson(pre.reshape(-1, 256, 256), targets.reshape(-1, 256, 256))
        a_pear = pearson_correlation(pre.reshape(-1, 256, 256), targets.reshape(-1, 256, 256))
        a_mse = mse(pre.reshape(-1, 256, 256), targets.reshape(-1, 256, 256))

        print(f'mse:{a_mse}; insu:{a_insu[0]}; pear:{a_pear[0]}')
        print(f'max:{np.max(targets)},min:{np.min(targets)}')
        print(f'max:{np.max(pre)},min:{np.min(pre)}')
        # if a_insu[0] > 0.85 and a_pear[0] > 0.85:
        print(f'max:{np.max(targets)},min:{np.min(targets)}')
        plot = MatrixPlot(output_path, targets, 'targets', chrom, seq_start,"Orca")
        plot.plot()

        print(f'max:{np.max(pre)},min:{np.min(pre)}')
        plot = MatrixPlot(output_path, pre, 'pre', chrom, seq_start,"Orca")
        plot.plot()
        insu_result[seq_start] = a_insu
        pear_resulr[seq_start] = a_pear
        start_result[seq_start] = seq_start


    for start, i_insu, i_pear in zip(start_result.values(), insu_result.values(), pear_resulr.values()):
        print(f"seq_start:{start};insu:{i_insu};pear:{i_pear}")