import trimesh
import numpy as np
import pandas as pd
from tqdm import tqdm

import config

def process_objects():

    # load raw metadata
    df = pd.read_csv(config.RAW_META_PATH)
    
    # fix class and object_path
    df["class"]       = df["object_id"].apply(lambda x: "_".join(x.split("_")[:-1]))
    df["object_path"] = df.apply(lambda row: f"{config.DATA_DIR}/{row['class']}/{row['split']}/{row['object_id']}.off", axis=1)
    df["points_path"] = df["object_path"].apply(lambda x: x.replace(".off", ".npy"))

    # load objects and save point clouds for fast training
    for _, row in tqdm(df.iterrows(), desc="Processing objects...", total=len(df)):
        points = trimesh.load_mesh(row["object_path"]).vertices
        np.save(row["points_path"], points, allow_pickle=True)

    # class labels as integers
    df["label"] = df["class"].astype("category").cat.codes
    
    # save processed metadata
    df.to_csv(config.PRC_META_PATH, index=False)

if __name__ == "__main__":
    process_objects()