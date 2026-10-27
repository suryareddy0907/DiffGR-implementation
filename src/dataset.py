"""
Dataset loader and preprocessor for DiffGR replication.
Supports ML-1M, Yelp2018, and Pinterest.
Performs minimal pre-processing (filter users < 5 interactions) and
creates 70% train / 10% validation / 20% test random splits.
"""

import os
import random
from collections import defaultdict
import numpy as np
import scipy.sparse as sp
import torch
from torch.utils.data import Dataset, DataLoader


class RecDataset(Dataset):
    """
    PyTorch Dataset yielding (user_id, interaction_vector) for training.
    """
    def __init__(self, n_users, n_items, user_train_dict):
        self.n_users = n_users
        self.n_items = n_items
        self.user_train_dict = user_train_dict
        self.users = list(range(n_users))

    def __len__(self):
        return self.n_users

    def __getitem__(self, idx):
        u = self.users[idx]
        items = self.user_train_dict.get(u, [])
        # Binary interaction vector (dense float tensor for the user's sub-graph G^0)
        vec = torch.zeros(self.n_items, dtype=torch.float32)
        if len(items) > 0:
            vec[items] = 1.0
        return torch.tensor(u, dtype=torch.long), vec


class DataLoaderHandler:
    def __init__(self, data_dir='Data', dataset_name='ml-1m', min_interactions=5, seed=42):
        self.data_dir = data_dir
        self.dataset_name = dataset_name.lower()
        self.min_interactions = min_interactions
        self.seed = seed
        
        self.user2id = {}
        self.item2id = {}
        self.n_users = 0
        self.n_items = 0
        
        self.train_dict = defaultdict(list)
        self.val_dict = defaultdict(list)
        self.test_dict = defaultdict(list)
        
        self._load_and_split()

    def _load_and_split(self):
        random.seed(self.seed)
        np.random.seed(self.seed)
        
        interactions = []
        
        if self.dataset_name in ['ml-1m', 'movielens-1m']:
            # Load ML-1M ratings.dat: UserID::MovieID::Rating::Timestamp
            path = os.path.join(self.data_dir, 'ml-1m', 'ml-1m', 'ratings.dat')
            if not os.path.exists(path):
                path = os.path.join(self.data_dir, 'ml-1m', 'ratings.dat')
            with open(path, 'r', encoding='latin-1') as f:
                for line in f:
                    parts = line.strip().split('::')
                    if len(parts) >= 2:
                        interactions.append((parts[0], parts[1]))

        elif self.dataset_name in ['pinterest', 'pinterest-20']:
            # Load Pinterest train & test rating files
            for fname in ['pinterest-20.train.rating', 'pinterest-20.test.rating']:
                path = os.path.join(self.data_dir, 'Pinterest', fname)
                with open(path, 'r', encoding='utf-8') as f:
                    for line in f:
                        parts = line.strip().split('\t')
                        if len(parts) >= 2:
                            interactions.append((parts[0], parts[1]))

        elif self.dataset_name in ['yelp', 'yelp2018']:
            # Load Yelp2018 train.txt and test.txt (NGCF format: user_id item_id1 item_id2 ...)
            for fname in ['train.txt', 'test.txt']:
                path = os.path.join(self.data_dir, 'Yelp2018', fname)
                with open(path, 'r', encoding='utf-8') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) > 1:
                            u = parts[0]
                            for item in parts[1:]:
                                interactions.append((u, item))
        else:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")

        # Group by user to filter users with fewer than min_interactions
        user_items_raw = defaultdict(set)
        for u, i in interactions:
            user_items_raw[u].add(i)

        filtered_users = {u: list(items) for u, items in user_items_raw.items() 
                          if len(items) >= self.min_interactions}

        # Build consistent 0-indexed ID mappings
        self.user2id = {u: idx for idx, u in enumerate(sorted(filtered_users.keys()))}
        all_items = set()
        for items in filtered_users.values():
            all_items.update(items)
        self.item2id = {i: idx for idx, i in enumerate(sorted(all_items))}

        self.n_users = len(self.user2id)
        self.n_items = len(self.item2id)

        # Randomly split each user's interactions: 70% train, 10% validation, 20% test
        for raw_u, items in filtered_users.items():
            u_id = self.user2id[raw_u]
            mapped_items = [self.item2id[item] for item in items]
            random.shuffle(mapped_items)

            n_inter = len(mapped_items)
            n_train = max(1, int(round(n_inter * 0.70)))
            n_val = max(1, int(round(n_inter * 0.10))) if n_inter >= 10 else 1
            if n_train + n_val >= n_inter:
                n_val = max(0, n_inter - n_train - 1)
            
            train_items = mapped_items[:n_train]
            val_items = mapped_items[n_train:n_train + n_val]
            test_items = mapped_items[n_train + n_val:]
            if len(test_items) == 0 and len(val_items) > 1:
                test_items = [val_items.pop()]

            self.train_dict[u_id] = train_items
            self.val_dict[u_id] = val_items
            self.test_dict[u_id] = test_items

        total_train = sum(len(v) for v in self.train_dict.values())
        total_val = sum(len(v) for v in self.val_dict.values())
        total_test = sum(len(v) for v in self.test_dict.values())
        total = total_train + total_val + total_test
        sparsity = (1.0 - total / (self.n_users * self.n_items)) * 100

        print(f"[{self.dataset_name.upper()}] Loaded {self.n_users:,} users, {self.n_items:,} items")
        print(f"[{self.dataset_name.upper()}] Total Interactions: {total:,} (Train: {total_train:,}, Val: {total_val:,}, Test: {total_test:,})")
        print(f"[{self.dataset_name.upper()}] Graph Sparsity: {sparsity:.2f}% | Potential Edges: {self.n_users * self.n_items:,}")

    def get_train_loader(self, batch_size=256, shuffle=True):
        dataset = RecDataset(self.n_users, self.n_items, self.train_dict)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)
