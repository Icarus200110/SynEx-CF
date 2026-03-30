import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import faiss
import time

class RACF:

    def __init__(self, database_path, k=10, n_candidates=5000, n_precise=1000, database_gene_order=None):
        self.database_path = database_path
        self.k = k
        self.n_candidates = n_candidates
        self.n_precise = n_precise
        self.database_gene_order_path = database_gene_order
        self.database = None
        self.database_vectors = None
        self.database_gene_order = None
        self.spot_vectors = None
        self.spot_gene_order = None
        self.shared_genes = None
        self.canonical_gene_order = None
        self.pca = None
        self.faiss_index = None

    def load_gene_order(self, gene_order_path):
        if gene_order_path is None:
            return None
        gene_order = []
        with open(gene_order_path, 'r') as f:
            for line in f:
                gene = line.strip()
                if gene:
                    gene_order.append(gene)
        return gene_order

    def align_gene_order(self, data, data_gene_order, target_gene_order):
        if data_gene_order is None or target_gene_order is None:
            return data

        if data_gene_order == target_gene_order:
            return data

        gene_to_idx = {gene: idx for idx, gene in enumerate(data_gene_order)}

        common_genes = []
        for gene in target_gene_order:
            if gene in gene_to_idx:
                common_genes.append(gene)

        if len(common_genes) == 0:
            raise ValueError("没有共同的基因，请检查基因顺序文件是否正确")

        if len(common_genes) != len(target_gene_order):
            print(f"警告: 只有 {len(common_genes)}/{len(target_gene_order)} 个基因匹配")

        aligned_data = np.zeros((data.shape[0], len(target_gene_order)))

        for new_idx, gene in enumerate(target_gene_order):
            if gene in gene_to_idx:
                old_idx = gene_to_idx[gene]
                aligned_data[:, new_idx] = data[:, old_idx]

        return aligned_data

    def load_data(self, spot_data=None, spot_gene_order=None):
        print("加载数据库数据...")
        self.database = pd.read_csv(self.database_path, sep='\t')

        self.database_gene_order = self.load_gene_order(self.database_gene_order_path)

        if self.database_gene_order is None:
            self.database_vectors = self.database.iloc[:, 1:].values
            self.database_gene_order = self.database.columns[1:].tolist()
        else:
            if len(self.database_gene_order) != self.database.shape[1] - 1:
                raise ValueError(f"数据库基因顺序文件中的基因数量 ({len(self.database_gene_order)}) 与数据库列数 ({self.database.shape[1] - 1}) 不匹配")
            self.database_vectors = self.database[self.database_gene_order].values

        if spot_data is not None:
            print("加载Spot数据...")
            if isinstance(spot_data, str):
                self.spot_data = pd.read_csv(spot_data, sep='\t')
            else:
                self.spot_data = spot_data

            self.spot_gene_order = self.load_gene_order(spot_gene_order)

            if self.spot_gene_order is None:
                self.spot_vectors = self.spot_data.iloc[:, 1:].values
                self.spot_gene_order = self.spot_data.columns[1:].tolist()
            else:
                self.spot_vectors = self.spot_data[self.spot_gene_order].values

    def preprocess(self):
        print("开始数据预处理...")

        if self.database_gene_order is None:
            self.database_gene_order = self.database.columns[1:].tolist()

        if self.spot_gene_order is None and hasattr(self, 'spot_data'):
            self.spot_gene_order = self.spot_data.columns[1:].tolist()

        if self.spot_gene_order is not None and self.database_gene_order is not None:
            self.shared_genes = list(set(self.database_gene_order) & set(self.spot_gene_order))
            print(f"共同基因数量: {len(self.shared_genes)}")

            if len(self.shared_genes) == 0:
                raise ValueError("数据库和Spot数据没有共同的基因，请检查基因顺序文件")

            self.canonical_gene_order = self.shared_genes

            print("执行基因顺序对齐...")
            self.database_vectors = self.align_gene_order(
                self.database_vectors, self.database_gene_order, self.canonical_gene_order
            )
            self.database_gene_order = self.canonical_gene_order

            self.spot_vectors = self.align_gene_order(
                self.spot_vectors, self.spot_gene_order, self.canonical_gene_order
            )
            self.spot_gene_order = self.canonical_gene_order

        print("执行标准化处理...")

        self.database_vectors = np.nan_to_num(self.database_vectors, nan=0.0, posinf=0.0, neginf=0.0)

        if hasattr(self, 'spot_vectors') and self.spot_vectors is not None:
            self.spot_vectors = np.nan_to_num(self.spot_vectors, nan=0.0, posinf=0.0, neginf=0.0)

        print("预处理完成！")
        print(f"数据库向量形状: {self.database_vectors.shape}")
        if hasattr(self, 'spot_vectors') and self.spot_vectors is not None:
            print(f"Spot向量形状: {self.spot_vectors.shape}")

    def build_index(self):
        print("构建FAISS索引...")
        d = self.database_vectors.shape[1]
        self.faiss_index = faiss.IndexFlatL2(d)
        self.faiss_index.add(self.database_vectors.astype('float32'))
        print("FAISS索引构建完成！")

    def calculate_similarity(self, spot_vector, database_vectors):
        def weighted_cosine_similarity(v1, v2):
            weights = np.std(database_vectors, axis=0)
            weights = weights / np.sum(weights)

            v1_weighted = v1 * weights
            v2_weighted = v2 * weights

            dot_product = np.dot(v1_weighted, v2_weighted.T)
            norm_v1 = np.linalg.norm(v1_weighted)
            norm_v2 = np.linalg.norm(v2_weighted, axis=1)

            return dot_product / (norm_v1 * norm_v2 + 1e-8)

        def jaccard_similarity(v1, v2):
            v1_bin = (v1 > np.percentile(v1, 80)).astype(int)
            v2_bin = (v2 > np.percentile(v2, 80, axis=1, keepdims=True)).astype(int)

            intersection = np.sum(v1_bin * v2_bin, axis=1)
            union = np.sum(v1_bin | v2_bin, axis=1)

            return intersection / (union + 1e-8)

        cosine_sim = weighted_cosine_similarity(spot_vector, database_vectors)
        jaccard_sim = jaccard_similarity(spot_vector, database_vectors)

        composite_sim = 0.7 * cosine_sim + 0.3 * jaccard_sim

        return composite_sim

    def generate_recommendations(self, spot_data=None, calculate_final_expression=False):
        if spot_data is not None:
            self.load_data(spot_data)
            self.preprocess()
            self.build_index()

        recommendations = {}
        final_expressions = []

        print(f"为每个spot生成 {self.k} 个推荐...")

        for i, spot_vec in enumerate(self.spot_vectors):
            if i % 100 == 0:
                print(f"处理第 {i} 个spot...")

            spot_vec_float32 = spot_vec.astype('float32').reshape(1, -1)
            distances, candidate_indices = self.faiss_index.search(spot_vec_float32, self.n_candidates)
            candidate_indices = candidate_indices[0]

            candidate_vectors = self.database_vectors[candidate_indices]

            similarities = self.calculate_similarity(spot_vec, candidate_vectors)

            precise_indices = np.argsort(similarities)[-self.n_precise:][::-1]
            precise_candidates = candidate_indices[precise_indices]
            precise_similarities = similarities[precise_indices]

            n_clusters = min(2 * self.k, 100)
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(self.database_vectors[precise_candidates])

            cluster_representatives = {}
            for idx, cluster in enumerate(cluster_labels):
                if cluster not in cluster_representatives or precise_similarities[idx] > cluster_representatives[cluster][1]:
                    cluster_representatives[cluster] = (precise_candidates[idx], precise_similarities[idx])

            representatives = list(cluster_representatives.values())
            representatives.sort(key=lambda x: x[1], reverse=True)

            if len(representatives) < self.k:
                all_candidates = list(zip(precise_candidates, precise_similarities))
                all_candidates.sort(key=lambda x: x[1], reverse=True)

                selected_ids = {rep[0] for rep in representatives}
                for candidate in all_candidates:
                    if candidate[0] not in selected_ids:
                        representatives.append(candidate)
                        selected_ids.add(candidate[0])
                        if len(representatives) >= self.k:
                            break

            final_recommendations = [rep[0] for rep in representatives[:self.k]]
            recommendations[i] = final_recommendations

            if calculate_final_expression:
                final_expr = self.calculate_final_expression(final_recommendations)
                final_expressions.append(final_expr)

        print("推荐生成完成！")

        if calculate_final_expression:
            return recommendations, np.array(final_expressions)
        else:
            return recommendations

    def calculate_final_expression(self, recommended_indices, method='weighted_average'):
        recommended_vectors = self.database_vectors[recommended_indices]

        if method == 'average':
            final_expr = np.mean(recommended_vectors, axis=0)
        elif method == 'weighted_average':
            final_expr = np.mean(recommended_vectors, axis=0)
        elif method == 'median':
            final_expr = np.median(recommended_vectors, axis=0)
        else:
            final_expr = np.mean(recommended_vectors, axis=0)

        return final_expr

    def run(self, spot_data, spot_gene_order=None, calculate_final_expression=False):
        start_time = time.time()

        self.load_data(spot_data, spot_gene_order)

        self.preprocess()

        self.build_index()

        results = self.generate_recommendations(None, calculate_final_expression)

        end_time = time.time()
        print(f"总运行时间: {end_time - start_time:.2f}秒")

        return results


if __name__ == "__main__":
    recommender = RACF(
        database_path="merged_data/merged_gene_expression.tsv",
        database_gene_order="merged_data/gene_order.txt",
        k=10
    )

    spot_data_path = "merged_data/merged_gene_expression.tsv"
    spot_gene_order_path = "merged_data/gene_order.txt"
    recommendations = recommender.run(spot_data_path, spot_gene_order_path)

    for spot_idx, rec_indices in recommendations.items():
        print(f"Spot {spot_idx} 的推荐: {rec_indices}")
        if spot_idx == 5:
            break
