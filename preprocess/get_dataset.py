import os
import torch
from torch.utils.data import Dataset
import numpy as np
from pysam import FastaFile
from skimage.transform import resize
from preprocess.data_feature import HiCFeature, DNAFeature, GenomicFeature

# 常量定义
BLOCK_SIZE = 2000000  # 2.5M
MARGIN = 5000000  # 5M


class GenomicDataset(Dataset):
    def __init__(self, fasta_path, hic_dir, genomic_path, mode='train',
                 windows=2097152, res=10000, output=256, genomic_features=False,use_aug=True):
        """
        初始化基因组数据集
        参数:
            fasta_path: 参考基因组的FASTA文件路径
            hic_dir: 存储样本HiC数据的目录
            genomic_path: 基因组特征文件路径
            mode: 数据集模式 ('train', 'valid', 'test')
            windows: 读取序列长度
            res: hic分辨率
            output: 输出矩阵大小
            genomic_features: 是否使用ATAC和CTCF特征
        """
        # 读取数据参数
        self.windows = windows
        self.res = res
        self.output = output
        self.mode = mode
        self.genomic_features = genomic_features

        # 存储路径信息
        self.fasta_path = fasta_path
        self.hic_dir = hic_dir
        self.atac_path = genomic_path + 'atac.bw'
        self.ctcf_path = genomic_path + 'ctcf_log2fc.bw'

        # 定义染色体分配
        # self.test_chroms = ['HC04_A05', 'HC04_D05']
        # self.valid_chroms = ['HC04_A06', 'HC04_D06', 'HC04_A07', 'HC04_D07']

        self.test_chroms = ['8', '9']
        self.valid_chroms = ['7']

        # self.test_chroms = ['chr8', 'chr9']
        # self.valid_chroms = ['chr7', 'chr10']

        # self.test_chroms = ['Sb_BTx623_Chr8', 'Sb_BTx623_Chr9']
        # self.valid_chroms = ['Sb_BTx623_Chr7']

        # 预加载染色体长度信息
        self.chrom_lengths = self._preload_chrom_lengths(fasta_path)

        # 生成样本位置
        self.entries = self._generate_samples()
        self.use_aug = use_aug
        # 创建的VCF特征提取器
        self.dna_feature = DNAFeature(path=fasta_path)
        if genomic_features:
            self.atac_feater = GenomicFeature(path=genomic_path + 'atac.bw', norm='log')
            self.ctcf_feater = GenomicFeature(path=genomic_path + 'h3k4me3.bw', norm='log')
            #self.h3k27m3_feater = GenomicFeature(path=genomic_path + 'h3k27me3.bw', norm='log')

        # 缓存HiC特征提取器
        self.hic_features = {}

        print(f"Initialized {mode} dataset with {len(self.entries)} samples")
        print(f"Data augmentation: {'Enabled' if use_aug else 'Disabled'}")

    def _generate_samples(self):
        """根据染色体划分生成样本位置"""
        entries = []

        # 获取所有染色体
        chroms = list(self.chrom_lengths.keys())

        for chrom in chroms:
            # 根据模式选择染色体
            if self.mode == 'test' and chrom not in self.test_chroms:
                continue
            if self.mode == 'valid' and chrom not in self.valid_chroms:
                continue
            if self.mode == 'train' and (chrom in self.test_chroms or chrom in self.valid_chroms):
                continue

            # 获取染色体长度
            chrom_length = self.chrom_lengths[chrom]

            # 跳过长度不足的染色体
            if chrom_length < 2 * MARGIN + self.windows:
                print(f"Skipping chromosome {chrom} (length {chrom_length} < {2 * MARGIN + self.windows})")
                continue

            # 计算有效区域
            start_pos = MARGIN
            end_pos = chrom_length - MARGIN

            # 划分采样区域
            current = start_pos
            while current + self.windows <= end_pos:
                # 计算中心位置
                entries.append({
                    'chrom': chrom,
                    'start': current,
                    'end': current + self.windows,
                })

                # 移动到下一个2.5M区块的起始位置
                current += BLOCK_SIZE

        return entries

    @staticmethod
    def _preload_chrom_lengths(fasta_path):
        """预加载所有染色体长度信息"""
        fasta = FastaFile(fasta_path)
        chrom_lengths = {chrom: length for chrom, length in zip(fasta.references, fasta.lengths)}
        fasta.close()
        return chrom_lengths

    def _get_hic_feature(self, chrom):
        """获取样本的HiC特征提取器（带缓存）"""
        cache_key = f"{chrom}"
        if cache_key not in self.hic_features:
            hic_path = os.path.join(self.hic_dir, f"Zm_B73_Chr{chrom}.npz")
            # hic_path = os.path.join(self.hic_dir, f"{chrom}.npz")
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
            start = block_start  # 固定位置

        end = start + self.windows

        # 获取DNA序列
        dna = self.dna_feature.get(chrom, start, end)
        #feature_tensor = torch.tensor(dna, dtype=torch.float32)  # [windows, 5]
        # 获取HiC矩阵
        hic_feature = self._get_hic_feature(chrom)
        hic_mat = hic_feature.get(start, window=self.windows, res=self.res)
        hic_mat = resize(hic_mat, (self.output, self.output), anti_aliasing=True)
        hic_mat = np.log(hic_mat + 1)


        if self.genomic_features:
            atac = self.atac_feater.get(chrom, start, end)
            ctcf = self.ctcf_feater.get(chrom, start, end)
            #h3k27me3 = self.h3k27m3_feater.get(chrom, start, end)
            features_list = [atac,ctcf]  
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
        """在区块内随机位移窗口位置"""
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
        """关闭所有打开的文件句柄"""
        self.dna_feature.close()
        if self.genomic_features:
            self.atac_feater.close()
            self.ctcf_feater.close()
            #self.h3k27m3_feater.close()
        for feature in self.hic_features.values():
            if hasattr(feature, 'close'):
                feature.close()


def collate_fn(batch):
    """批处理样本并跳过错误"""
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
