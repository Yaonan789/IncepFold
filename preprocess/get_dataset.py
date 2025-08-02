import os
import torch
from torch.utils.data import Dataset
import numpy as np
from pysam import FastaFile
from skimage.transform import resize
from preprocess.data_feature import HiCFeature, DNAFeature, GenomicFeature


BLOCK_SIZE = 2500000  # 2.5M
MARGIN = 5000000  # 5M


class GenomicDataset(Dataset):
    def __init__(self, fasta_path, hic_dir, genomic_path, mode='train',
                 windows=2097152, res=10000, output=256, genomic_features=False,use_aug=True):

       
        self.windows = windows
        self.res = res
        self.output = output
        self.mode = mode
        self.genomic_features = genomic_features

        
        self.fasta_path = fasta_path
        self.hic_dir = hic_dir
        
        self.h3k4_path = genomic_path + 'H3K4me3.bw'

   
        self.test_chroms = ['HC04_A05', 'HC04_D05']
        self.valid_chroms = ['HC04_A06', 'HC04_D06', 'HC04_A07', 'HC04_D07']


        self.chrom_lengths = self._preload_chrom_lengths(fasta_path)

   
        self.entries = self._generate_samples()
        self.use_aug = use_aug

        self.dna_feature = DNAFeature(path=fasta_path)
        if genomic_features:
            self.h3k4_feater = GenomicFeature(path=genomic_path + 'H3K4me3.bw', norm='log')


        self.hic_features = {}

        print(f"Initialized {mode} dataset with {len(self.entries)} samples")
        print(f"Data augmentation: {'Enabled' if use_aug else 'Disabled'}")

    def _generate_samples(self):

        entries = []

        chroms = list(self.chrom_lengths.keys())

        for chrom in chroms:

            if self.mode == 'test' and chrom not in self.test_chroms:
                continue
            if self.mode == 'valid' and chrom not in self.valid_chroms:
                continue
            if self.mode == 'train' and (chrom in self.test_chroms or chrom in self.valid_chroms):
                continue


            chrom_length = self.chrom_lengths[chrom]

            if chrom_length < 2 * MARGIN + self.windows:
                print(f"Skipping chromosome {chrom} (length {chrom_length} < {2 * MARGIN + self.windows})")
                continue


            start_pos = MARGIN
            end_pos = chrom_length - MARGIN


            current = start_pos
            while current + self.windows <= end_pos:

                entries.append({
                    'chrom': chrom,
                    'start': current,
                    'end': current + self.windows,
                })

                current += BLOCK_SIZE

        return entries

    @staticmethod
    def _preload_chrom_lengths(fasta_path):

        fasta = FastaFile(fasta_path)
        chrom_lengths = {chrom: length for chrom, length in zip(fasta.references, fasta.lengths)}
        fasta.close()
        return chrom_lengths

    def _get_hic_feature(self, chrom):

        cache_key = f"{chrom}"
        if cache_key not in self.hic_features:
            hic_path = os.path.join(self.hic_dir, f"{chrom}.npz")
            if not os.path.exists(hic_path):
                raise FileNotFoundError(f"HiC file not found: {hic_path}")
            self.hic_features[cache_key] = HiCFeature(path=hic_path)
        return self.hic_features[cache_key]

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        pos_entry = self.entries[idx]

        chrom = pos_entry['chrom']
        block_start = pos_entry['start']
        block_end = pos_entry['end']

        if self.use_aug and self.mode == 'train':
            start = self.shift_aug(block_start, block_end)
        else:
            start = block_start 

        end = start + self.windows

        dna = self.dna_feature.get(chrom, start, end)

        hic_feature = self._get_hic_feature(chrom)
        hic_mat = hic_feature.get(start, window=self.windows, res=self.res)
        hic_mat = resize(hic_mat, (self.output, self.output), anti_aliasing=True)
        hic_mat = np.log(hic_mat + 1)


        if self.genomic_features:

            h3k4 = self.h3k4_feater.get(chrom, start, end)
            features_list = [h3k4]
            if self.use_aug and self.mode == 'train':
                dna = self.gaussian_noise(dna, 0.1)
                # Genomic features
                features_list = [self.gaussian_noise(item, 0.1) for item in features_list]
                # Reverse complement all data
                dna, features_list, hic_mat = self.reverse(dna, hic_mat, features_list)
            combined_features = np.concatenate((dna, np.array(features_list).T), axis=1)
        else:
            if self.use_aug and self.mode == 'train':
                dna = self.gaussian_noise(dna, 0.1)
                dna, _, hic_mat = self.reverse(dna, hic_mat, None)
            combined_features = dna

        feature_tensor = torch.tensor(combined_features, dtype=torch.float32)
        #print(f'feature_tensor.shape:{feature_tensor.shape}')
        hic_tensor = torch.tensor(hic_mat, dtype=torch.float32)  # [output, output]
        #print(f'feature_tensor.shape:{hic_tensor.shape}')
        return feature_tensor, hic_tensor

    def shift_aug(self, block_start, block_end):

        max_shift = block_end - block_start - self.windows
        if max_shift > 0:
            shift = np.random.randint(0, max_shift)
            return block_start + shift
        return block_start

    def gaussian_noise(self, inputs, std=1.0):
        noise = np.random.randn(*inputs.shape) * std
        outputs = inputs + noise
        return outputs

    def reverse(self, seq, mat, features=None, chance=0.5):
        '''
        Reverse sequence and matrix
        '''
        r_bool = np.random.rand(1)
        features_r = None
        if r_bool < chance:
            seq_r = np.flip(seq, 0).copy()  # n x 5 shape
            if features != None:
                features_r = [np.flip(item, 0).copy() for item in features]  # n
            mat_r = np.flip(mat, [0, 1]).copy()  # n x n

            # Complementary sequence
            seq_r = self.complement(seq_r)
        else:
            seq_r = seq
            if features != None:
                features_r = features
            mat_r = mat
        return seq_r, features_r, mat_r

    def complement(self, seq, chance=0.5):
        '''
        Complimentary sequence
        '''
        r_bool = np.random.rand(1)
        if r_bool < chance:
            seq_comp = np.concatenate([seq[:, 3:4],
                                       seq[:, 2:3],
                                       seq[:, 1:2],
                                       seq[:, 0:1],
                                       seq[:, 4:5]], axis=1)
        else:
            seq_comp = seq
        return seq_comp


    def close(self):

        self.dna_feature.close()
        if self.genomic_features:
            self.h3k4_feater.close()
        for feature in self.hic_features.values():
            if hasattr(feature, 'close'):
                feature.close()


def collate_fn(batch):

    dna_batch = []
    hic_batch = []

    for item in batch:
        if item is None:
            continue
        dna, hic = item
        dna_batch.append(dna)
        hic_batch.append(hic)

    if len(dna_batch) == 0:
        return None, None

    dna_tensors = torch.stack(dna_batch)
    hic_tensors = torch.stack(hic_batch)
    return dna_tensors, hic_tensors