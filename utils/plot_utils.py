import os
import numpy as np
import pandas as pd

class MatrixPlot:
    # MatrixPlot(output_path, targets, 'targets', chrom, seq_start)
    def __init__(self, output_path, image, prefix, chr_name, start_pos,model_name):
        self.output_path = output_path,
        self.prefix = prefix
        self.chr_name = chr_name
        self.start_pos = start_pos
        self.model_name = model_name
        self.create_save_path(output_path, prefix)
        self.image = self.preprocess_image(image)

    def get_enhanced_colormap(self):
        from matplotlib.colors import LinearSegmentedColormap
        # High-contrast red-white gradient
        colors = [
            (1, 1, 1),        # White
            (1, 0.9, 0.9),    # Light pink
            (0.9, 0.6, 0.6),  # Pink
            (0.8, 0.3, 0.3),  # Red
            (0.7, 0.1, 0.1),  # Dark red
            (0.5, 0, 0)       # Deep red
        ]
        return LinearSegmentedColormap.from_list("enhanced_red", colors, N=256)

    def create_save_path(self, output_path, prefix):
        self.save_path = f'{output_path}/{prefix}'
        os.makedirs(f'{self.save_path}/imgs', exist_ok = True)
        os.makedirs(f'{self.save_path}/npy', exist_ok = True)

    def preprocess_image(self, image):
        return image

    def plot(self, vmin=None, vmax=None, contrast_boost=1.5):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 7))  # Increase canvas size

        # Dynamically calculate color range
        data = self.image.copy()
        if vmin is None:
            vmin = np.percentile(data, 5)  # Use 5th percentile instead of min value
        if vmax is None:
            vmax = np.percentile(data, 95) * contrast_boost  # Use 95th percentile and enhance contrast

        # Create high-contrast color map
        color_map = self.get_enhanced_colormap()

        # Add colorbar
        im = ax.imshow(data, cmap=color_map, aspect='equal',
                       vmin=vmin, vmax=vmax, interpolation='nearest')
        fig.colorbar(im, ax=ax, label='Interaction Frequency')

        self.reformat_ticks(plt)

    def reformat_ticks(self, plt):
        # Rescale tick labels
        current_ticks = np.arange(0, 250, 50)
        plt.xticks(current_ticks, self.rescale_coordinates(current_ticks, self.start_pos))
        plt.yticks(current_ticks, self.rescale_coordinates(current_ticks, self.start_pos))
        # Format labels
        plt.ylabel('Genomic position (Mb)')
        
        if self.prefix=='targets':
            plt.xlabel(f'Target Hi-C Matrix: {self.chr_name} {self.start_pos} - {self.start_pos + 2097152}')
        else:
            plt.xlabel(f'{self.model_name} Predicted Hi-C Matrix: {self.chr_name} {self.start_pos} - {self.start_pos + 2097152}')
        self.save_data(plt)

    def rescale_coordinates(self, coords, zero_position):
        scaling_ratio = 8192
        replaced_coords = coords * scaling_ratio + zero_position
        coords_mb = replaced_coords / 1000000
        str_list = [f'{item:.2f}' for item in coords_mb]
        return str_list

    def save_data(self, plt):
        plt.savefig(f'{self.save_path}/imgs/{self.chr_name}_{self.start_pos}.png', bbox_inches='tight')
        plt.close()
        np.save(f'{self.save_path}/npy/{self.chr_name}_{self.start_pos}', self.image)
