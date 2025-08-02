# IncepFold

A Multimodal Inception-Based Model for 3D Genome Architecture Prediction in Cotton

## Usage

### Data

- All necessary data used in this study—including DNA sequences, H3K4me3, and both raw and processed Hi-C data—are available at https://zenodo.org/records/16716764.
- Please place the DNA sequences files in `/data/genome/cotton`.
- Please place the H3K4me3 files in `/data/genomic_features/cotton`.
- Please place the Hi-C files in `/data/hic/cotton`.

### Running

Our experiments were conducted using a single NVIDIA L20 GPU. To reproduce the results, please use a single GPU setup for training, testing, and plotting.

- Run `main.py` for training.
- Run `test.py` for evaluation.
- Run `plot.py` for visualization.

## Finally

Thank you for reading our work. We sincerely hope it can be helpful to you!