import sys
sys.path.append('../src')

import numpy as np
import pytest
from algs.searcher import NumpySearcher  # 确保替换成包含NumpySearcher类的模块名

def test_initialization():
    searcher = NumpySearcher(feat_len=128)
    assert searcher.feat_len == 128
    assert searcher.gallery is None
    assert isinstance(searcher.idx_2_uuid, list) and len(searcher.idx_2_uuid) == 0

def test_add_features_and_uuids():
    searcher = NumpySearcher(feat_len=256)
    features = np.random.rand(10, 256)
    uuids = [f"id_{i}" for i in range(10)]
    searcher.add(features, uuids=uuids)

    assert searcher.gallery.shape == (10, 256)
    assert len(searcher.idx_2_uuid) == 10
    assert searcher.idx_2_uuid == uuids

def test_search_by_topk():
    searcher = NumpySearcher(feat_len=256)
    features = np.random.rand(10, 256)
    uuids = [f"id_{i}" for i in range(10)]
    searcher.add(features, uuids=uuids)

    query = np.random.rand(3, 256)
    topk_scores, topk_idxs, topk_uuids = searcher.search_by_topk(query, topk=5)

    assert topk_scores.shape == (3, 5)
    assert topk_idxs.shape == (3, 5)
    assert all(len(row) == 5 for row in topk_uuids)

    # Ensure that topk UUIDs are correctly mapped
    for i, idx_row in enumerate(topk_idxs):
        for j, idx in enumerate(idx_row):
            assert topk_uuids[i][j] == uuids[idx]

def test_add_invalid_uuids():
    searcher = NumpySearcher(feat_len=256)
    features = np.random.rand(5, 256)
    uuids = [f"id_{i}" for i in range(4)]  # Intentionally wrong length

    with pytest.raises(ValueError):
        searcher.add(features, uuids=uuids)

def test_feature_dimension_mismatch():
    searcher = NumpySearcher(feat_len=256)
    # Generate features with incorrect dimension
    incorrect_features = np.random.rand(5, 255)

    with pytest.raises(ValueError):
        searcher.add(incorrect_features)

    # Test query dimension mismatch
    incorrect_query = np.random.rand(1, 255)
    with pytest.raises(ValueError):
        searcher.search_by_topk(incorrect_query, topk=5)

# Optionally, configure pytest to run all tests when this file is executed directly
if __name__ == "__main__":
    pytest.main()
