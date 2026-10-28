"""Retrieval-Augmented Collaborative Filtering on paired core/full references."""
from pathlib import Path
import numpy as np
import pandas as pd
import faiss
from sklearn.cluster import KMeans


class RACF:
    def __init__(self, database_path, k=10, n_candidates=5000, n_precise=1000,
                 database_gene_order=None, tau=0.1, nlist=100, pq_m=16, nbits=8,
                 nprobe=10, training_sections=None, validation_sections=(),
                 test_sections=(), reference_sections=None, seed=42):
        if min(k, n_candidates, n_precise, nlist, pq_m, nbits, nprobe) < 1 or tau <= 0:
            raise ValueError('RACF counts and temperature must be positive.')
        if k > n_precise or n_precise > n_candidates:
            raise ValueError('Require cluster count <= reranked K <= coarse candidate count.')
        self.database_path = database_path
        self.database_gene_order_path = database_gene_order
        self.k, self.n_candidates, self.n_precise = k, n_candidates, n_precise
        self.tau, self.nlist, self.pq_m = tau, nlist, pq_m
        self.nbits, self.nprobe, self.seed = nbits, nprobe, seed
        self.training_sections = None if training_sections is None else set(training_sections)
        self.excluded_sections = set(validation_sections) | set(test_sections)
        if self.training_sections is not None and self.training_sections & self.excluded_sections:
            raise ValueError('Reference training sections overlap held-out sections.')
        self.reference_sections = reference_sections
        self.faiss_index = None

    @staticmethod
    def load_gene_order(value):
        if value is None:
            return None
        if isinstance(value, (list, tuple, np.ndarray)):
            return list(value)
        return [line.strip() for line in Path(value).read_text(encoding='utf-8').splitlines()
                if line.strip()]

    @staticmethod
    def _read(value):
        return pd.read_csv(value, sep='\t') if isinstance(value, (str, Path)) else value.copy()

    @staticmethod
    def _matrix(frame, genes):
        if len(set(genes)) != len(genes) or not set(genes).issubset(frame.columns):
            raise ValueError('Gene identifiers must be unique and present in the expression matrix.')
        array = frame[genes].to_numpy(dtype=np.float32)
        if not np.isfinite(array).all():
            raise ValueError('RACF expression must be finite and already library/log2 normalized.')
        return np.ascontiguousarray(array)

    def load_data(self, spot_data=None, spot_gene_order=None):
        self.database = self._read(self.database_path)
        if self.database.columns.duplicated().any():
            raise ValueError('Duplicate database columns.')
        if 'section' in self.database:
            sections = self.database['section'].astype(str).to_numpy()
        elif self.reference_sections is not None:
            sections = np.asarray(self.reference_sections, dtype=str)
            if len(sections) != len(self.database):
                raise ValueError('Reference section metadata does not match database rows.')
        else:
            raise ValueError('RACF requires reference section provenance for held-out exclusion.')
        if self.training_sections is None:
            raise ValueError('Explicit training_sections are required to construct the reference.')
        keep = np.isin(sections, list(self.training_sections))
        keep &= ~np.isin(sections, list(self.excluded_sections))
        self.database = self.database.loc[keep].reset_index(drop=True)
        if len(self.database) < 2:
            raise ValueError('IVF-PQ requires at least two permitted reference profiles.')
        metadata = {'section', 'spot_id', 'patient', self.database.columns[0]}
        self.database_gene_order = (self.load_gene_order(self.database_gene_order_path)
                                    or [c for c in self.database if c not in metadata])
        # FULL measured expression remains separate from the 785-dimensional retrieval vectors.
        self.full_gene_order = list(self.database_gene_order)
        self.full_expression = self._matrix(self.database, self.full_gene_order)
        if spot_data is not None:
            self.spot_data = self._read(spot_data)
            metadata = {'section', 'spot_id', 'patient', self.spot_data.columns[0]}
            self.spot_gene_order = (self.load_gene_order(spot_gene_order)
                                   or [c for c in self.spot_data if c not in metadata])
            self.spot_vectors = self._matrix(self.spot_data, self.spot_gene_order)
        self.faiss_index = None

    def preprocess(self):
        if len(self.spot_gene_order) != 785:
            raise ValueError('RACF queries must contain the 785 directly predicted genes.')
        self.canonical_gene_order = list(self.spot_gene_order)
        if not set(self.canonical_gene_order).issubset(self.full_gene_order):
            raise ValueError('Every core query gene must also belong to the full reference output.')
        self.shared_genes = list(self.canonical_gene_order)
        self.database_vectors = self._matrix(self.database, self.canonical_gene_order)
        # Do not standardize, variance-weight, or alter cosine geometry.
        self.spot_vectors = np.ascontiguousarray(self.spot_vectors, dtype=np.float32)

    def _pad(self, vectors):
        return np.ascontiguousarray(np.pad(vectors, ((0, 0), (0, self.index_dim - vectors.shape[1]))),
                                    dtype=np.float32)

    def build_index(self):
        n, d = self.database_vectors.shape
        # 785 is not divisible by usual PQ subquantizer counts. Zero padding preserves L2.
        self.index_dim = ((d + self.pq_m - 1) // self.pq_m) * self.pq_m
        bits = min(self.nbits, int(np.floor(np.log2(n))))
        lists = min(self.nlist, max(1, n // 40))
        quantizer = faiss.IndexFlatL2(self.index_dim)
        self.faiss_index = faiss.IndexIVFPQ(quantizer, self.index_dim, lists, self.pq_m, bits)
        self.faiss_index.cp.seed = self.seed
        self.faiss_index.pq.cp.seed = self.seed
        references = self._pad(self.database_vectors)
        self.faiss_index.train(references)
        self.faiss_index.add(references)
        self.faiss_index.nprobe = min(self.nprobe, lists)

    @staticmethod
    def calculate_similarity(spot_vector, database_vectors):
        denominator = (np.linalg.norm(spot_vector)
                       * np.linalg.norm(database_vectors, axis=1)).clip(min=1e-12)
        return (database_vectors @ spot_vector) / denominator

    def calculate_final_expression(self, recommended_indices, method='weighted_average',
                                   query_vector=None):
        if method != 'weighted_average':
            raise ValueError('The manuscript uses temperature-weighted full reference profiles.')
        if query_vector is None:
            raise ValueError('Query-dependent cosine weights require the current core prediction.')
        similarities = self.calculate_similarity(
            query_vector, self.database_vectors[recommended_indices])
        logits = similarities / self.tau
        weights = np.exp(logits - logits.max())
        weights /= weights.sum()
        return weights @ self.full_expression[recommended_indices]

    def generate_recommendations(self, spot_data=None, calculate_final_expression=False):
        if spot_data is not None:
            return self.run(spot_data, calculate_final_expression=calculate_final_expression)
        if self.faiss_index is None:
            raise ValueError('Call load_data, preprocess and build_index first.')
        recommendations, expressions = {}, []
        for i, query in enumerate(self.spot_vectors):
            _, coarse = self.faiss_index.search(self._pad(query[None]),
                                               min(self.n_candidates, len(self.database_vectors)))
            candidates = np.unique(coarse[0][coarse[0] >= 0])
            if not len(candidates):
                raise RuntimeError('IVF-PQ found no candidates; increase nprobe.')
            scores = self.calculate_similarity(query, self.database_vectors[candidates])
            selected = candidates[np.argsort(-scores, kind='stable')[:self.n_precise]]
            vectors = self.database_vectors[selected]
            clusters = min(self.k, len(np.unique(vectors, axis=0)))
            kmeans = KMeans(n_clusters=clusters, random_state=self.seed, n_init=10).fit(vectors)
            representatives = []
            for cluster in range(clusters):
                members = np.flatnonzero(kmeans.labels_ == cluster)
                distances = ((vectors[members] - kmeans.cluster_centers_[cluster]) ** 2).sum(1)
                representatives.append(int(selected[members[np.argmin(distances)]]))
            recommendations[i] = representatives
            if calculate_final_expression:
                expressions.append(self.calculate_final_expression(representatives, query_vector=query))
        return ((recommendations, np.asarray(expressions, dtype=np.float32))
                if calculate_final_expression else recommendations)

    def run(self, spot_data, spot_gene_order=None, calculate_final_expression=False):
        self.load_data(spot_data, spot_gene_order)
        self.preprocess()
        self.build_index()
        return self.generate_recommendations(calculate_final_expression=calculate_final_expression)

