# IncepFold

[IncepFold: A Deep Learning Framework for 3D Genome Prediction in Diverse Plant Species](https://link.springer.com/chapter/10.1007/978-981-92-0369-7_1)

🎉🎉🎉 Our work has been accepted as a **Full Paper** at **DASFAA 2026**!

## Usage

### Data

* Please refer to the paper for data acquisition.
* Please place the DNA sequence files in a directory such as `/data/genome/cotton`.
* Please place the H3K4me3 and ATAC files in a directory such as `/data/genomic_features/cotton`.
* Please place the Hi-C files in a directory such as `/data/hic/cotton`.

### Running

Our experiments were conducted using a single NVIDIA L20 GPU. To reproduce the results, please use a single GPU setup for training, testing, and plotting.

* Run `main.py` for training.
* Run `test.py` for evaluation.
* Run `plot.py` for visualization.

## Citation

If you find this work useful, please consider citing our paper:

```bibtex
@inproceedings{li2026incepfold,
  title     = {IncepFold: A Deep Learning Framework for 3D Genome Prediction in Diverse Plant Species},
  author    = {Li, Lu and Shen, Hao and Liu, Xinyuan and Deng, Pengcheng and Wang, Zhengchang and Wang, Maojun and Zhang, Zeyu},
  booktitle = {Database Systems for Advanced Applications},
  series    = {Lecture Notes in Computer Science},
  volume    = {16537},
  pages     = {3--18},
  year      = {2026},
  publisher = {Springer},
  address   = {Singapore},
  doi       = {10.1007/978-981-92-0369-7_1}
}
```
