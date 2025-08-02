import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import time
import matplotlib.pyplot as plt
from tqdm import tqdm

###############################################
from preprocess.get_dataset import GenomicDataset, collate_fn
from model.IncepFold import IncepFold
from metrics.metrics import insulation_pearson,mse,pearson_correlation,observed_vs_expected,distance_stratified_correlation

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)


class Config:
    # Configuration parameters
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    use_aug = True
    genomic_features = True
    model = IncepFold().to(device)
    species = 'cotton'
    windows = 2097152
    res = 10000
    output = 256

    # File paths
    fasta_path = f'data/genome/{species}/l128.chr.fa'
    genomic_path = f'data/genomic_features/{species}/'
    hic_dir = f"data/hic/{species}/"

    # Training parameters
    batch_size = 4
    learning_rate = 2e-4
    weight_decay = 1e-5
    epochs = 100
    patience = 20  # For early stopping

    # Output directories
    model_dir = f"output/saved_models/{species}/{model.__class__.__name__}_{genomic_features}/"
    os.makedirs(model_dir, exist_ok=True)
    best_model_path = os.path.join(model_dir, "best_model.pth")

    log_dir = f"output/logs/{species}/{model.__class__.__name__}_{genomic_features}/"
    os.makedirs(log_dir, exist_ok=True)
    results_file = os.path.join(log_dir, "training_results.txt")
    plot_file = os.path.join(log_dir, "training_plot.png")
    plot_dis_path = os.path.join(log_dir, "val_dis_plot.png")

config = Config()


def train_epoch(model, dataloader, criterion, optimizer, device):
    # Train model for one epoch
    model.train()
    running_loss = 0.0
    all_preds = []
    all_targets = []
    total_samples = 0

    progress_bar = tqdm(dataloader, desc="Training")
    for batch_idx, (dna, hic) in enumerate(progress_bar):
        dna = dna.to(device)
        hic_target = hic.to(device)

        optimizer.zero_grad()
        outputs = model(dna)
        loss = criterion(outputs, hic_target)
        loss.backward()
        optimizer.step()

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

    all_preds_array = np.array(all_preds)
    all_targets_array = np.array(all_targets)

    # Calculate evaluation metrics
    avg_insu = np.nanmean(insulation_pearson(all_preds_array, all_targets_array))  
    avg_mse = np.nanmean(mse(all_preds_array, all_targets_array))
    avg_pear = np.nanmean(pearson_correlation(all_preds_array, all_targets_array))
    avg_oe = np.nanmean(observed_vs_expected(all_preds_array, all_targets_array))
    epoch_loss = running_loss / total_samples
    return epoch_loss, avg_insu, avg_mse, avg_pear, avg_oe


def validate_epoch(model, dataloader, criterion, device):
    # Evaluate model on validation data
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []
    total_samples = 0

    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc="Validation")
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

    all_preds_array = np.array(all_preds)
    all_targets_array = np.array(all_targets)

    # Calculate validation metrics
    avg_insu = np.nanmean(insulation_pearson(all_preds_array, all_targets_array))  
    avg_mse = np.nanmean(mse(all_preds_array, all_targets_array))
    avg_pear = np.nanmean(pearson_correlation(all_preds_array, all_targets_array))
    avg_oe = np.nanmean(observed_vs_expected(all_preds_array, all_targets_array))
    avg_dis = np.nanmean(distance_stratified_correlation(all_preds_array, all_targets_array),axis=0)
    epoch_loss = running_loss / total_samples

    return epoch_loss, avg_insu, avg_mse, avg_pear, avg_oe, avg_dis


def main():
    # Load training and validation datasets
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
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=collate_fn)

    # Initialize model, loss, optimizer, and scheduler
    model = config.model
    print(f'config.device:{config.device}')
    print(f"Initial Model :{config.model.__class__.__name__} !!!")
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

    # Initialize training logs
    train_losses, train_insu_corrs, train_mses, train_pears, train_oes = [], [], [], [], []
    val_losses, val_insu_corrs, val_mses, val_pears, val_oes = [], [], [], [], []

    best_val_loss = float('inf')
    baes_val_dises = []
    epochs_without_improvement = 0

    # Write training results header
    with open(config.results_file, 'w') as f:
        f.write("Epoch\tTrain Loss\tVal Loss\tTrain insu Corr\tVal insu Corr\tTrain Mse\tVal Mse\tTrain Pearson\tVal Pearson\t"
                "Train OE\tVal OE\tTime\n")

    start_time = time.time()
    print(f'################ Start train!!!! ######################')
    for epoch in range(config.epochs):
        epoch_start = time.time()

        print(f"\nEpoch {epoch + 1}/{config.epochs}")
        print("-" * 50)

        # Training step
        train_loss, train_insu_corr, train_mse, train_pear, train_oe = train_epoch(
            model, train_loader, criterion, optimizer, config.device
        )

        # Validation step
        val_loss, val_insu_corr, val_mse, val_pear, val_oe, val_dis = validate_epoch(
            model, val_loader, criterion, config.device
        )

        # Logging metrics
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

        epoch_time = time.time() - epoch_start

        # Print epoch summary
        print(f"\nEpoch {epoch + 1} Summary:")
        print(f"  Train Loss: {train_loss:.8f}, Val Loss: {val_loss:.8f}")
        print(f"  Train Mse: {train_mse:.8f}, Val Mse: {val_mse:.8f}")
        print(f"  Train Insu corrs: {train_insu_corr:.4f}, Val Insu corrs: {val_insu_corr:.4f}")
        print(f"  Train Pearson: {train_pear:.4f}, Val Pearson: {val_pear:.4f}")
        print(f"  Train Observed vs expected: {train_oe:.4f}, Val Observed vs expected: {val_oe:.4f}")
        print(f"  Time: {epoch_time:.2f} seconds")

        # Save metrics to log file
        with open(config.results_file, 'a') as f:
            f.write(
                f"{epoch + 1}\t{train_loss:.8f}\t{val_loss:.8f}\t{train_insu_corr:.4f}\t{val_insu_corr:.4f}\t"
                f"{train_mse:.8f}\t{val_mse:.8f}\t{train_pear:.4f}\t{val_pear:.4f}\t"
                f"{train_oe:.4f}\t{val_oe:.4f}\t{epoch_time:.2f}\n")

        scheduler.step(val_loss)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            baes_val_dises = val_dis
            epochs_without_improvement = 0

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

        if epochs_without_improvement >= config.patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    # Print training summary
    total_time = time.time() - start_time
    print(f"\nTraining complete in {total_time:.2f} seconds")
    print(f"Best validation loss: {best_val_loss:.8f}")
    print(f"Best validation Mse: {val_mse:.8f}")
    print(f"Best validation Insu correlation: {val_insu_corr:.4f}")
    print(f"Best validation Pearson correlation: {val_pear:.4f}")
    print(f"Best validation Observed vs expected: {val_oe:.4f}")

    # Plot and save training curves
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.plot(train_insu_corrs, label='Train Pearson')  
    plt.plot(val_insu_corrs, label='Validation Pearson')
    plt.xlabel('Epochs')
    plt.ylabel('Pearson Correlation')
    plt.title('Training and Validation Pearson Correlation')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(config.plot_file)
    plt.close()

    # Plot distance-stratified correlation
    plt.figure(figsize=(12, 8))
    plt.plot(baes_val_dises, marker='o', linestyle='-', color='b')
    plt.title('baes_val_dises')
    plt.xlabel('The position from the diagonal')
    plt.ylabel('Pearson Correlation')
    plt.grid(True)
    plt.savefig(config.plot_dis_path)

    # Clean up resources
    train_dataset.close()
    val_dataset.close()


if __name__ == "__main__":
    main()
