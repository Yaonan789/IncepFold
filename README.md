# IncepFold

A Multimodal Inception-Based Model for 3D Genome Architecture Prediction in Cotton

## Usage

### Data

- The DNA sequences can be downloaded from https://zenodo.org/records/16686495.
- The processed H3K4me3 data is included in this repository.
- The processed Hi-C data is also included. If you would like to access the raw Hi-C files, please download them from https://zenodo.org/records/16686495.

### Running

Our experiments were conducted using a single NVIDIA L20 GPU. To reproduce the results, please use a single GPU setup for training, testing, and plotting.

- Run `main.py` for training.
- Run `test.py` for evaluation.
- Run `plot.py` for visualization.

## Finally

Thank you for reading our work. We sincerely hope it can be helpful to you!