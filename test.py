import os
import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from metrics.metrics import insulation_pearson, pearson_correlation, mse, observed_vs_expected, distance_stratified_correlation

from model.IncepFold import IncepFold
from preprocess.get_dataset import GenomicDataset, collate_fn

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

class Config:
    # Configuration class to hold model and dataset parameters
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    use_aug = True
    genomic_features = True
    model= IncepFold().to(device)  # Initialize the model
    species = 'cotton'
    windows = 2097152

    res = 10000
    output = 256

    fasta_path = f'data/genome/{species}/l128.chr.fa'
    genomic_path = f'data/genomic_features/{species}/'
    hic_dir = f"data/hic/{species}/"

    batch_size = 4

    model_path = f'output/saved_models/{species}/{model.__class__.__name__}_{genomic_features}/best_model.pth'
    log_dir = f"output/logs/{species}/{model.__class__.__name__}_{genomic_features}/test/"
    os.makedirs(log_dir, exist_ok=True)
    results_file = os.path.join(log_dir, "test_results.txt")
    plot_dis_path = os.path.join(log_dir, "test_dis_plot.png")

config = Config()

def test_epoch(model, dataloader, criterion, device):
    # Function to evaluate the model for one epoch on the test set
    model.eval()
    running_loss = 0.0
    all_preds = []  
    all_targets = []  
    total_samples = 0

    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc="Test")
        for batch_idx, (dna, hic) in enumerate(progress_bar):

            dna = dna.to(device)
            hic_target = hic.to(device)

            outputs = model(dna)
            loss = criterion(outputs, hic_target)

            batch_size = dna.size(0)
            batch_loss = loss.item()
            running_loss += batch_loss * batch_size
            total_samples += batch_size

            all_preds.extend(outputs.detach().cpu().numpy())
            all_targets.extend(hic_target.detach().cpu().numpy())

            avg_loss = running_loss / total_samples

            progress_bar.set_postfix({
                'batch_loss': f"{batch_loss:.8f}",
                'avg_loss': f"{avg_loss:.8f}"
            })

    if total_samples == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    # Convert predictions and targets to numpy arrays for metric calculation
    all_preds_array = np.array(all_preds)  
    all_targets_array = np.array(all_targets)  

    avg_insu = np.nanmean(insulation_pearson(all_preds_array, all_targets_array))  
    avg_mse = np.nanmean(mse(all_preds_array, all_targets_array))
    avg_pear = np.nanmean(pearson_correlation(all_preds_array, all_targets_array))
    avg_oe = np.nanmean(observed_vs_expected(all_preds_array, all_targets_array))
    avg_dis = np.nanmean(distance_stratified_correlation(all_preds_array, all_targets_array), axis=0)
    epoch_loss = running_loss / total_samples

    return epoch_loss, avg_insu, avg_mse, avg_pear, avg_oe, avg_dis

if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    genomic_features = True
    species = 'cotton'
    model = config.model

    # Load the trained model checkpoint
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

    # Initialize test dataset
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

    # Initialize dataloader
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )

    criterion = nn.MSELoss()  # Mean squared error loss

    # Run test
    test_loss, test_insu_corr, test_mse, test_pear, test_oe, test_dis = test_epoch(
        model, test_loader, criterion, config.device
    )

    # Print evaluation metrics
    print(f"Best validation loss: {test_loss:.8f}")
    print(f"Best validation Mse: {test_mse:.8f}")
    print(f"Best validation Insu correlation: {test_insu_corr:.4f}")
    print(f"Best validation Pearson correlation: {test_pear:.4f}")
    print(f"Best validation Observed vs expected: {test_oe:.4f}")

    # Plot distance-stratified correlation
    plt.figure(figsize=(12, 8))
    plt.plot(test_dis, marker='o', linestyle='-', color='b')
    plt.title('baes_val_dises')
    plt.xlabel('The position from the diagonal')
    plt.ylabel('Pearson Correlation')
    plt.grid(True)
    plt.savefig(config.plot_dis_path)

    test_dataset.close()
