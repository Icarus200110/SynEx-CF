"""Tissue ontology and geometric helpers shared by training and prediction."""
from src.contracts import LABEL2ID


def create_label_vocabulary(data_dir=None):
    # The breast ontology is declared in the manuscript; never infer it from held-out labels.
    mapping = dict(LABEL2ID)
    return mapping, {i: name for name, i in mapping.items()}, len(mapping)


def get_neighbor_coordinates(x, y, grid_size=1):
    return [(x + dx, y + dy) for dy in range(-grid_size, grid_size + 1)
            for dx in range(-grid_size, grid_size + 1)]


def create_spatial_context(spot_id, label_map, default_idx=-1):
    x, y = map(int, spot_id.split('x'))
    return [label_map.get(f'{nx}x{ny}', default_idx)
            for nx, ny in get_neighbor_coordinates(x, y)]

