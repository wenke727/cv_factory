import numpy as np
from metric.similarity import cosine_similarity

def normalize(feature):
    return feature / (np.linalg.norm(feature) + 1e-12)

class GallerySearcher:
    def __init__(self, feat_len=256):
        self.feat_len = feat_len
        self.gallery = None
        self.idx_2_uuid = []

    def add(self, features, uuids=None):
        """ Add features to the gallery along with corresponding UUIDs """
        features = np.array(features, dtype=np.float32)

        # Check if features have the correct dimensions
        if features.shape[1] != self.feat_len:
            raise ValueError(f"All features must be of length {self.feat_len}")

        features = normalize(features)

        if self.gallery is None:
            self.gallery = features
        else:
            self.gallery = np.vstack([self.gallery, features])

        if uuids is not None:
            if len(uuids) != len(features):
                raise ValueError("The length of UUIDs must match the number of features added.")
            self.idx_2_uuid.extend(uuids)
        else:
            self.idx_2_uuid.extend(range(len(self.idx_2_uuid), len(self.idx_2_uuid) + len(features)))

    def search_by_topk(self, query, topk=3):
        """ Search the top k closest vectors in the gallery for each query vector """
        query = np.array(query, dtype=np.float32)
        if query.ndim == 1:
            query = query[np.newaxis, :]  # Make query two-dimensional

        if query.shape[1] != self.feat_len:
            raise ValueError(f"All queries must be of length {self.feat_len}")

        query = normalize(query)
        scores = np.dot(query, self.gallery.T)

        topk_idxs = np.argsort(-scores, axis=1)[:, :topk]
        topk_scores = np.take_along_axis(scores, topk_idxs, axis=1)

        # Map indices to UUIDs
        topk_uuids = [[self.idx_2_uuid[idx] for idx in row] for row in topk_idxs]

        return topk_scores, topk_idxs, topk_uuids

    def search_best(self, query, similarity_thres=.3):
        """
        Searches the gallery for the best match above a specified similarity threshold.

        Args:
            query (np.ndarray): The query feature vector or array of vectors.
            similarity_thres (float): The minimum cosine similarity threshold required to consider
                                      a match valid.

        Returns:
            tuple: A tuple containing the highest similarity score, the corresponding index,
                   and the UUID of the best match if the score exceeds the threshold. If no match
                   exceeds the threshold, returns (0, None, None).
        """
        if not isinstance(query, np.ndarray):
            return 0, None, None

        query = np.array(query, dtype=np.float32)
        if query.ndim == 1:
            query = query[np.newaxis, :]  # Make query two-dimensional

        if self.gallery is not None:
            sim = cosine_similarity(query, self.gallery)[0]
            cand_idx = sim.argmax()

            if sim[cand_idx] > similarity_thres:
                return sim[cand_idx], cand_idx, self.idx_2_uuid[cand_idx]

        return 0, None, None

    def __repr__(self):
        return f"NumpySearcher: {str(self.idx_2_uuid)}"


class FlatSearcher:
    def __init__(self, ngpu=1, feat_len=256):
        import faiss
        if ngpu:
            flat_config = []
            for i in range(ngpu):
                cfg = faiss.GpuIndexFlatConfig()
                cfg.useFloat16 = False
                cfg.device = i
                flat_config.append(cfg)
            res = [faiss.StandardGpuResources() for _ in range(ngpu)]
            indexes = [
                faiss.GpuIndexFlatIP(res[i], feat_len, flat_config[i])
                for i in range(ngpu)
            ]
            self.index = faiss.IndexProxy()
            for sub_index in indexes:
                self.index.addIndex(sub_index)
        else:
            self.index = faiss.IndexFlatL2(feat_len)

    def search_by_topk(self, query, gallery, topk=16):
        self.index.reset()
        self.index.add(gallery)
        topk_scores, topk_idxs = self.index.search(query, topk)
        return topk_scores, topk_idxs


# Example usage
if __name__ == "__main__":
    searcher = GallerySearcher()
    features = np.random.rand(10, 256)  # Random gallery features
    uuids = ['id1', 'id2', 'id3', 'id4', 'id5', 'id6', 'id7', 'id8', 'id9', 'id10']
    searcher.add(features, uuids=uuids)

    query = np.random.rand(256)  # Random query features
    topk_scores, topk_idxs, topk_uuids = searcher.search_by_topk(query, topk=3)
    print("Top-k scores:", topk_scores)
    print("Top-k indices:", topk_idxs)
    print("Top-k UUIDs:", topk_uuids)

