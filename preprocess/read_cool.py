import numpy as np
from cooler import Cooler
import cooler
import os

def main(path, save_path, resolution, window_size, balance=True):
    hic = Cooler(f'{path}')
    cooler.balance_cooler(hic, store=True)
    data = hic.matrix(balance=balance, sparse=True)
    print(f'has store balance!')
    for chrom in hic.chromnames:
        mat = data.fetch(chrom)
        diags = compress_diag(mat, window_size)
        ucsc_chrom = f'{chrom}_{resolution}_{window_size}.npz'
        chrom_path = f'{save_path}/{ucsc_chrom}'
        os.makedirs(f'{save_path}/', exist_ok=True) 
        np.savez(chrom_path, **diags)
        print(f'npz has saved {chrom_path}')

# Do not use balanced matrix

def compress_diag(mat, window):
    # NOTE: dict is probably suboptimal here. We could have a big list double the window_size
    diag_dict = {}
    for d in range(window):
        diag_dict[str(d)] = np.nan_to_num(mat.diagonal(d).astype(np.half))
        diag_dict[str(-d)] = np.nan_to_num(mat.diagonal(-d).astype(np.half))
    return diag_dict

if __name__ == '__main__':

    path = f'L128.10000.cool'
    save_path = f'hic'
    resolution = 10000
    window_size = 256
    main(path, save_path, resolution, window_size)

    print(f'All sample has processed!!')

