import numpy as np
from pysam import FastaFile
import pyBigWig as pbw
import os


class Feature:
    """特征基类，定义通用接口"""

    def load(self, **kwargs):
        """加载资源，子类必须实现"""
        raise NotImplementedError('load method not implemented')

    def get(self, *args, **kwargs):
        """获取数据，子类必须实现"""
        raise NotImplementedError('get method not implemented')

    def __len__(self):
        """返回资源数量，子类必须实现"""
        raise NotImplementedError('__len__ method not implemented')

    def close(self):
        """释放资源，子类可选实现"""
        pass

    def __enter__(self):
        """支持上下文管理器"""
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """退出上下文时自动关闭资源"""
        self.close()


class DNAFeature(Feature):
    """DNA序列特征处理器"""

    def __init__(self, path):
        """
        初始化DNA序列处理器

        Args:
            path (str): FASTA文件路径
        """
        self.path = path
        self.fasta = None
        self.chrom_lengths = {}
        self.chroms = []
        self._load(path)

    def _load(self, path):
        """加载FASTA文件并验证"""
        if not os.path.exists(path):
            raise FileNotFoundError(f"FASTA file not found: {path}")

        try:
            self.fasta = FastaFile(path)
            self.chrom_lengths = {k: v for k, v in zip(self.fasta.references, self.fasta.lengths)}
            self.chroms = list(self.fasta.references)
        except Exception as e:
            raise IOError(f"Failed to load FASTA file: {path}\nError: {str(e)}")

    def get(self, chrom, start, end, **kwargs):
        """
        获取指定区域的DNA序列(one-hot编码)

        Args:
            chrom (str): 染色体名称
            start (int): 起始位置
            end (int): 结束位置

        Returns:
            np.ndarray: one-hot编码序列 (L, 5)
        """
        seq = self.get_seq(chrom, start, end)
        return self.onehot_encode(seq)

    def get_seq(self, chrom, start, end):
        """
        获取指定区域的DNA序列(整数编码)

        Args:
            chrom (str): 染色体名称
            start (int): 起始位置
            end (int): 结束位置

        Returns:
            np.ndarray: 整数编码序列 (L,)
        """
        # 验证坐标有效性
        self._validate_coordinates(chrom, start, end)

        # 获取并编码序列
        seq = self.fasta.fetch(chrom, start, end).upper()
        en_dict = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 4}
        return np.array([en_dict.get(ch, 4) for ch in seq], dtype=np.int8)

    def _validate_coordinates(self, chrom, start, end):
        """验证基因组坐标有效性"""
        if chrom not in self.chrom_lengths:
            raise ValueError(f"Chromosome {chrom} not found in FASTA")

        chrom_length = self.chrom_lengths[chrom]
        if start < 0 or end > chrom_length:
            raise IndexError(f"Coordinates {start}-{end} out of range (0-{chrom_length})")
        if start >= end:
            raise ValueError(f"Start ({start}) must be less than end ({end})")

    def read_all_chrom(self):
        """获取所有染色体名称列表"""
        return self.chroms.copy()

    @staticmethod
    def onehot_encode(seq):
        """
        将整数序列编码为one-hot矩阵

        Args:
            seq (np.ndarray): 整数编码序列

        Returns:
            np.ndarray: one-hot矩阵 (L, 5)
        """
        seq_emb = np.zeros((len(seq), 5), dtype=np.float32)
        seq_emb[np.arange(len(seq)), seq] = 1
        return seq_emb

    def __len__(self):
        """返回染色体数量"""
        return len(self.chroms)

    def close(self):
        """安全关闭文件资源"""
        if self.fasta is not None:
            self.fasta.close()
            self.fasta = None

    def __repr__(self):
        return f"DNAFeature(path='{self.path}', chroms={len(self.chroms)})"


class GenomicFeature(Feature):
    """基因组特征处理器（支持bigWig文件）"""

    def __init__(self, path, norm=None):
        """
        初始化基因组特征处理器

        Args:
            path (str): bigWig文件路径
            norm (str, optional): 归一化方法 ('log' 或 None)
        """
        self.path = path
        self.norm = norm
        self.bw_file = None
        self.chrom_lengths = {}
        self.chroms = []
        self._load(path)

    def _load(self, path):
        """加载bigWig文件并验证"""
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
        """
        获取指定区域的基因组特征值

        Args:
            chr_name (str): 染色体名称
            start (int): 起始位置
            end (int): 结束位置

        Returns:
            np.ndarray: 特征值数组 (L,)
        """
        # 验证坐标有效性
        self._validate_coordinates(chr_name, start, end)

        # 读取信号值
        signals = self.bw_file.values(chr_name, start, end)
        feature = np.array(signals, dtype=np.float32)

        # 处理缺失值
        feature = np.nan_to_num(feature, nan=0.0)

        # 应用归一化
        return self._apply_normalization(feature)

    def _apply_normalization(self, data):
        """应用指定的归一化方法"""
        if self.norm == 'log':
            data = np.log(data + 1)  # log(1+x) 数值更稳定
            return data
        elif self.norm is None or self.norm == '':
            return data
        else:
            raise ValueError(f'Unsupported normalization type: {self.norm}')

    def _validate_coordinates(self, chr_name, start, end):
        """验证基因组坐标有效性"""
        if chr_name not in self.chrom_lengths:
            raise ValueError(f"Chromosome {chr_name} not found in bigWig")

        chrom_length = self.chrom_lengths[chr_name]
        if start < 0 or end > chrom_length:
            raise IndexError(f"Coordinates {start}-{end} out of range (0-{chrom_length})")
        if start >= end:
            raise ValueError(f"Start ({start}) must be less than end ({end})")

    def length(self, chr_name):
        """获取指定染色体的长度"""
        return self.chrom_lengths.get(chr_name, 0)

    def __len__(self):
        """返回染色体数量"""
        return len(self.chroms)

    def close(self):
        """安全关闭文件资源"""
        if self.bw_file is not None:
            self.bw_file.close()
            self.bw_file = None

    def __repr__(self):
        return f"GenomicFeature(path='{self.path}', norm='{self.norm}', chroms={len(self.chroms)})"


class HiCFeature(Feature):
    """Hi-C接触矩阵处理器"""

    def __init__(self, path):
        """
        初始化Hi-C处理器

        Args:
            path (str): NPZ文件路径
        """
        self.path = path
        self.hic = None
        self._load(path)

    def _load(self, path):
        """加载Hi-C数据文件并验证"""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Hi-C file not found: {path}")
        try:
            print(f'Loading Hi-C data: {path}')
            self.hic = dict(np.load(path))
            # 验证数据格式
            if '0' not in self.hic:
                raise ValueError("Invalid Hi-C format: missing '0' diagonal")
        except Exception as e:
            raise IOError(f"Failed to load Hi-C file: {path}\nError: {str(e)}")

    def get(self, start, window=2000000, res=10000):
        """
        获取指定区域的Hi-C接触矩阵

        Args:
            start (int): 起始位置
            window (int): 窗口大小（默认2Mb）
            res (int): 分辨率（默认10kb）

        Returns:
            np.ndarray: Hi-C接触矩阵 (bins, bins)
        """
        start_bin = int(start / res)
        range_bin = int(window / res)
        end_bin = start_bin + range_bin

        # 验证边界
        max_bin = len(self.hic['0'])
        if end_bin > max_bin:
            raise IndexError(f"Requested bins {start_bin}-{end_bin} exceed max {max_bin}")

        return self._diag_to_mat(start_bin, end_bin)

    def _diag_to_mat(self, start, end):
        """
        从对角线数据重建接触矩阵

        Args:
            start (int): 起始bin
            end (int): 结束bin

        Returns:
            np.ndarray: 接触矩阵 (L, L)
        """
        square_len = end - start
        diag_load = {}

        # 收集对角线数据
        for diag_i in range(square_len):
            # 正对角线
            diag_key = str(diag_i)
            if diag_key in self.hic:
                data = self.hic[diag_key][start: start + square_len - diag_i]
                diag_load[diag_key] = data

            # 负对角线
            neg_key = str(-diag_i)
            if neg_key in self.hic:
                data = self.hic[neg_key][start: start + square_len - diag_i]
                diag_load[neg_key] = data

        # 重建矩阵
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
        """返回主对角线长度"""
        return len(self.hic['0']) if '0' in self.hic else 0

    def close(self):
        """Hi-C数据在内存中，无需显式关闭"""
        pass

    def __repr__(self):
        return f"HiCFeature(path='{self.path}', bins={len(self)})"


def safe_execute(func, *args, **kwargs):
    """安全执行函数并处理异常"""
    try:
        return func(*args, **kwargs)
    except Exception as e:
        print(f"Error in {func.__name__}: {str(e)}")
        return None


