import os
import numpy as np
import torch
from metrics.metrics import insulation_pearson, pearson_correlation, mse
from model.IncepFold import IncepFold
from preprocess.data_feature import HiCFeature, DNAFeature, GenomicFeature
from utils.plot_utils import MatrixPlot
from skimage.transform import resize

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

if __name__ == '__main__':
    # Set device to GPU if available
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    genomic_features = True
    species = 'cotton'

    # Load model and its checkpoint
    model = IncepFold().to(device)
    model_name = model.__class__.__name__
    checkpoint = torch.load(f'output/saved_models/{species}/{model_name}_{genomic_features}/best_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print("Epoch:", checkpoint['epoch'])
    print("Val Loss:", checkpoint['val_loss'])
    print("Val Pearson:", checkpoint['val_pear'])
    print("val_insu_corr:", checkpoint['val_insu_corr'])

    chrom = 'HC04_A05'
    windows = 2097152  # Input window size (2M)
    res = 10000        # Resolution for Hi-C
    output = 256       # Output resolution for visualization

    # Dictionaries to store results
    start_result = {}
    insu_result = {}
    pear_resulr = {}

    # Iterate over genome segments
    for seq_start in range(0, 109051905, 2097152):
        print(f'seq_start:{seq_start}')
        seq_end = seq_start + windows

        # Define paths to genome and genomic features
        fasta_path = f'data/genome/{species}/l128.chr.fa'
        genomic_path = f'data/genomic_features/{species}/'
        hic_dir = f"data/hic/{species}/"
        path = hic_dir + f'{chrom}.npz'
        output_path = f'output/hic_fig'
        os.makedirs(output_path, exist_ok=True)

        # Load DNA features
        dna_feature = DNAFeature(path=fasta_path)
        H3K4me3_feater = GenomicFeature(path=genomic_path + 'H3K4me3.bw', norm='log')

        # Get DNA one-hot encoding
        dna = dna_feature.get(chrom, seq_start, seq_end)
        deature_tensor = torch.tensor(dna, dtype=torch.float32)
        input_tensor = deature_tensor.unsqueeze(0).to(device)

        if genomic_features:
            # Get genomic feature (e.g., histone modification signal)
            H3K4me3 = H3K4me3_feater.get(chrom, seq_start, seq_end).reshape(-1, 1)
            print(f'H3K4me3 max:{np.max(H3K4me3)} H3K4me3 min{np.min(H3K4me3)}')

            # Concatenate DNA and genomic feature
            combined_features = np.concatenate((dna, H3K4me3), axis=1)
            input_tensor = torch.tensor(combined_features, dtype=torch.float32).unsqueeze(0).to(device)

        print(f"input_tensor.shape:{input_tensor.shape}")

        # Load Hi-C target matrix and apply preprocessing
        hic_feature = HiCFeature(path=path)
        targets = hic_feature.get(seq_start, windows, res)
        targets = resize(targets, (256, 256), anti_aliasing=True)
        targets = np.log(targets + 1)

        # Run model prediction
        pre = model(input_tensor)
        pre = pre.squeeze(0).detach().cpu().numpy()

        # Evaluate prediction
        a_insu = insulation_pearson(pre.reshape(-1, 256, 256), targets.reshape(-1, 256, 256))
        a_pear = pearson_correlation(pre.reshape(-1, 256, 256), targets.reshape(-1, 256, 256))
        a_mse = mse(pre.reshape(-1, 256, 256), targets.reshape(-1, 256, 256))

        print(f'mse:{a_mse}; insu:{a_insu[0]}; pear:{a_pear[0]}')
        print(f'max:{np.max(targets)},min:{np.min(targets)}')
        print(f'max:{np.max(pre)},min:{np.min(pre)}')

        # Plot ground truth and prediction Hi-C matrices
        plot = MatrixPlot(output_path, targets, 'targets', chrom, seq_start, model_name)
        plot.plot()

        plot = MatrixPlot(output_path, pre, 'pre', chrom, seq_start, model_name)
        plot.plot()

        # Save results
        insu_result[seq_start] = a_insu
        pear_resulr[seq_start] = a_pear
        start_result[seq_start] = seq_start

    # Print all collected results
    for start, i_insu, i_pear in zip(start_result.values(), insu_result.values(), pear_resulr.values()):
        print(f"seq_start:{start};insu:{i_insu};pear:{i_pear}")
