import numpy as np
from pysam import FastaFile
import pyBigWig as pbw
import os


class Feature:

    def load(self, **kwargs):
        raise NotImplementedError('load method not implemented')

    def get(self, *args, **kwargs):
        raise NotImplementedError('get method not implemented')

    def __len__(self):
        raise NotImplementedError('__len__ method not implemented')

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


class DNAFeature(Feature):

    def __init__(self, path):

        self.path = path
        self.fasta = None
        self.chrom_lengths = {}
        self.chroms = []
        self._load(path)

    def _load(self, path):

        if not os.path.exists(path):
            raise FileNotFoundError(f"FASTA file not found: {path}")

        try:
            self.fasta = FastaFile(path)
            self.chrom_lengths = {k: v for k, v in zip(self.fasta.references, self.fasta.lengths)}
            self.chroms = list(self.fasta.references)
        except Exception as e:
            raise IOError(f"Failed to load FASTA file: {path}\nError: {str(e)}")

    def get(self, chrom, start, end, **kwargs):
        seq = self.get_seq(chrom, start, end)
        return self.onehot_encode(seq)

    def get_seq(self, chrom, start, end):

        self._validate_coordinates(chrom, start, end)

        seq = self.fasta.fetch(chrom, start, end).upper()
        en_dict = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 4}
        return np.array([en_dict.get(ch, 4) for ch in seq], dtype=np.int8)

    def _validate_coordinates(self, chrom, start, end):

        if chrom not in self.chrom_lengths:
            raise ValueError(f"Chromosome {chrom} not found in FASTA")

        chrom_length = self.chrom_lengths[chrom]
        if start < 0 or end > chrom_length:
            raise IndexError(f"Coordinates {start}-{end} out of range (0-{chrom_length})")
        if start >= end:
            raise ValueError(f"Start ({start}) must be less than end ({end})")

    def read_all_chrom(self):

        return self.chroms.copy()

    @staticmethod
    def onehot_encode(seq):

        seq_emb = np.zeros((len(seq), 5), dtype=np.float32)
        seq_emb[np.arange(len(seq)), seq] = 1
        return seq_emb

    def __len__(self):

        return len(self.chroms)

    def close(self):

        if self.fasta is not None:
            self.fasta.close()
            self.fasta = None

    def __repr__(self):
        return f"DNAFeature(path='{self.path}', chroms={len(self.chroms)})"


class GenomicFeature(Feature):

    def __init__(self, path, norm=None):

        self.path = path
        self.norm = norm
        self.bw_file = None
        self.chrom_lengths = {}
        self.chroms = []
        self._load(path)

    def _load(self, path):

        if not os.path.exists(path):
            raise FileNotFoundError(f"BigWig file not found: {path}")

        try:
            self.bw_file = pbw.open(path)
            self.chroms = list(self.bw_file.chroms().keys())
            self.chrom_lengths = {chrom: self.bw_file.chroms(chrom) for chrom in self.chroms}
            print(f'Loaded genomic feature: {path} | Normalization: {self.norm}')
        except Exception as e:
            raise IOError(f"Failed to load bigWig file: {path}\nError: {str(e)}")

    def get(self, chr_name, start, end):

        self._validate_coordinates(chr_name, start, end)


        signals = self.bw_file.values(chr_name, start, end)
        feature = np.array(signals, dtype=np.float32)
        
        feature = np.nan_to_num(feature, nan=0.0)

        return self._apply_normalization(feature)

    def _apply_normalization(self, data):

        if self.norm == 'log':
            data = np.log(data + 1) 
            return data
        elif self.norm is None or self.norm == '':
            return data
        else:
            raise ValueError(f'Unsupported normalization type: {self.norm}')

    def _validate_coordinates(self, chr_name, start, end):

        if chr_name not in self.chrom_lengths:
            raise ValueError(f"Chromosome {chr_name} not found in bigWig")

        chrom_length = self.chrom_lengths[chr_name]
        if start < 0 or end > chrom_length:
            raise IndexError(f"Coordinates {start}-{end} out of range (0-{chrom_length})")
        if start >= end:
            raise ValueError(f"Start ({start}) must be less than end ({end})")

    def length(self, chr_name):
        return self.chrom_lengths.get(chr_name, 0)

    def __len__(self):
        return len(self.chroms)

    def close(self):
        if self.bw_file is not None:
            self.bw_file.close()
            self.bw_file = None

    def __repr__(self):
        return f"GenomicFeature(path='{self.path}', norm='{self.norm}', chroms={len(self.chroms)})"


class HiCFeature(Feature):

    def __init__(self, path):

        self.path = path
        self.hic = None
        self._load(path)

    def _load(self, path):

        if not os.path.exists(path):
            raise FileNotFoundError(f"Hi-C file not found: {path}")
        try:
            print(f'Loading Hi-C data: {path}')
            self.hic = dict(np.load(path))

            if '0' not in self.hic:
                raise ValueError("Invalid Hi-C format: missing '0' diagonal")
        except Exception as e:
            raise IOError(f"Failed to load Hi-C file: {path}\nError: {str(e)}")

    def get(self, start, window=2000000, res=10000):

        start_bin = int(start / res)
        range_bin = int(window / res)
        end_bin = start_bin + range_bin

        max_bin = len(self.hic['0'])
        if end_bin > max_bin:
            raise IndexError(f"Requested bins {start_bin}-{end_bin} exceed max {max_bin}")

        return self._diag_to_mat(start_bin, end_bin)

    def _diag_to_mat(self, start, end):

        square_len = end - start
        diag_load = {}

        for diag_i in range(square_len):

            diag_key = str(diag_i)
            if diag_key in self.hic:
                data = self.hic[diag_key][start: start + square_len - diag_i]
                diag_load[diag_key] = data


            neg_key = str(-diag_i)
            if neg_key in self.hic:
                data = self.hic[neg_key][start: start + square_len - diag_i]
                diag_load[neg_key] = data

        matrix = np.zeros((square_len, square_len), dtype=np.float32)
        for i in range(square_len):
            for j in range(square_len):
                diag_index = j - i
                diag_key = str(diag_index)

                if diag_key in diag_load:
                    pos = min(i, j) if diag_index >= 0 else min(i - diag_index, j)
                    if 0 <= pos < len(diag_load[diag_key]):
                        matrix[i, j] = diag_load[diag_key][pos]

        return matrix

    def __len__(self):

        return len(self.hic['0']) if '0' in self.hic else 0

    def close(self):
        pass

    def __repr__(self):
        return f"HiCFeature(path='{self.path}', bins={len(self)})"


def safe_execute(func, *args, **kwargs):
    try:
        return func(*args, **kwargs)
    except Exception as e:
        print(f"Error in {func.__name__}: {str(e)}")
        return None


