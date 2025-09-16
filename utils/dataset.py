import torch
import trimesh
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

import config

class PointDataset(Dataset):
    """
    Custom Dataset for training point cloud SetTransformer.
    """
    def __init__(self, split, augment):
        """
        Args:
            split    (str): Filter metadata by 'train', or 'test' split.
            augment (bool): Whether to apply random rotation and scaling.
        """
        # read processed metadata
        metadata = pd.read_csv(config.PRC_META_PATH)

        # filter by split
        self.metadata = metadata[metadata["split"] == split]

        self.augment = augment

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx: int):

        # filter rows by idx
        row = self.metadata.iloc[idx]

        # load pickled point cloud
        points = np.load(row["points_path"])

        # choose random subset of points
        num_pnts = config.TRAINING_PARAMS["num_pnts"]
        choice   = np.random.choice(len(points), num_pnts, replace=True)
        points   = points[choice,:]
        points   = self.standardize(points)

        # augment point cloud
        if self.augment:
            points = self.random_transform(points)

        points = torch.tensor(points).float()

        # load label
        label = torch.tensor(row["label"]).long()

        return points, label

    def standardize(self, x: np.ndarray):
        """
        Standardize coordinates of a batch of point clouds.
        """
        clipper = np.mean(abs(x), keepdims=True)
        z       = np.clip(x, -100 * clipper, 100 * clipper)
        mean    = np.mean(z)
        std     = np.std(z)
        return (z - mean) / std
    
    def random_transform(self, x: np.ndarray):
        """
        Applies a random rotation and uniform scaling to an array of points.
        """
        # Convert points to homogeneous coordinates (N, 4)
        x_homogeneous = np.hstack([x, np.ones((x.shape[0], 1))])

        # Create Rotation Matrix - random quaternion to 4x4 matrix
        random_quaternion = trimesh.transformations.random_quaternion()
        rotation_matrix   = trimesh.transformations.quaternion_matrix(random_quaternion)

        # Rotate in XY plane only
        rotation_matrix[2,2]   = 1
        rotation_matrix[0:2,2] = 0
        rotation_matrix[2,0:2] = 0

        # Create Scaling Matrix - random factor between 0.8 and 1.25
        scale_factor   = np.random.uniform(0.8, 1.25)
        scaling_matrix = trimesh.transformations.scale_matrix(scale_factor)

        # Combine the transformations
        transform_matrix = rotation_matrix @ scaling_matrix

        # Apply
        x_transformed = x_homogeneous @ transform_matrix
        
        # Convert to (N, 3)
        return x_transformed[:, :3]