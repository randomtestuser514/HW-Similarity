import os
import random
from itertools import combinations
from collections import defaultdict

def group_files_by_writer(data_dir, extension=('.tif', '.tf')):
    """
    Group files by writer id (first four digits of the filename).
    Returns a dict: {writer_id: [filepath1, filepath2, ...]}
    """
    groups = defaultdict(list)
    for filename in os.listdir(data_dir):
        if filename.lower().endswith(extension):
            writer_id = filename[:4]
            groups[writer_id].append(os.path.join(data_dir, filename))
    print(f"[Data] Grouped files into {len(groups)} writer groups.")
    return groups

def train_test_split(groups, train_ratio=0.8, seed=42):
    """
    Split the writer groups into training and testing based on writer id.
    Returns two dicts: (train_groups, test_groups)
    """
    writer_ids = list(groups.keys())
    random.seed(seed)
    random.shuffle(writer_ids)
    num_train = int(len(writer_ids) * train_ratio)
    train_ids = writer_ids[:num_train]
    test_ids = writer_ids[num_train:]
    
    train_groups = {wid: groups[wid] for wid in train_ids}
    test_groups = {wid: groups[wid] for wid in test_ids}
    print(f"[Data] Train groups: {len(train_groups)} | Test groups: {len(test_groups)}")
    return train_groups, test_groups

def create_pairs(groups, negatives_per_positive=1, seed=42):
    """
    Create pairs of images with labels.
      - Positive pairs: all unique combinations within the same writer (if at least 2 samples exist)
      - Negative pairs: for every positive pair, sample a given number of negative pairs by pairing images from different writers.
    
    Returns a list of tuples (path1, path2, label) where label=1 for same writer, 0 otherwise.
    """
    random.seed(seed)
    pairs = []
    
    # Positive pairs: for each writer with at least 2 images.
    positive_pairs = []
    for writer, files in groups.items():
        if len(files) < 2:
            continue
        for pair in combinations(files, 2):
            positive_pairs.append((pair[0], pair[1], 1))
    
    pairs.extend(positive_pairs)
    
    # Negative pairs: for each positive pair, sample negatives.
    writers = list(groups.keys())
    negative_pairs = []
    for (file1, file2, _) in positive_pairs:
        writer1 = os.path.basename(file1)[:4]
        possible_writers = [w for w in writers if w != writer1 and groups[w]]
        if possible_writers:
            for _ in range(negatives_per_positive):
                neg_writer = random.choice(possible_writers)
                neg_file = random.choice(groups[neg_writer])
                negative_pairs.append((file1, neg_file, 0))
    pairs.extend(negative_pairs)
    
    random.shuffle(pairs)
    print(f"[Data] Created {len(positive_pairs)} positive pairs and {len(negative_pairs)} negative pairs.")
    return pairs

def prepare_data(data_dir, train_ratio=0.8, negatives_per_positive=1, seed=42):
    """
    Given the data directory, split data into training and testing pairs.
    Returns two lists: train_pairs and test_pairs.
    """
    print("[Data] Preparing data...")
    groups = group_files_by_writer(data_dir)
    train_groups, test_groups = train_test_split(groups, train_ratio, seed)
    train_pairs = create_pairs(train_groups, negatives_per_positive, seed)
    test_pairs = create_pairs(test_groups, negatives_per_positive, seed)
    print("[Data] Data preparation complete.")
    return train_pairs, test_pairs

if __name__ == "__main__":
    data_directory = "./data"  # folder with your image files
    train_pairs, test_pairs = prepare_data(data_directory, train_ratio=0.8, negatives_per_positive=2)
    print("[Data] Number of training pairs:", len(train_pairs))
    print("[Data] Number of testing pairs:", len(test_pairs))
