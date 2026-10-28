"""Training-reference linear imputation for the no-RACF ablation."""
import numpy as np
from sklearn.linear_model import Ridge


class LinearImputer:
    """Map the same 785 predicted core genes to the training reference gene space."""
    def __init__(self, alpha=1.0):
        self.model = Ridge(alpha=alpha)

    def fit(self, training_dataset):
        self.core_genes = list(training_dataset.gene_set)
        self.full_gene_order = list(training_dataset.full_gene_order)
        reference = training_dataset.reference_frame()
        self.model.fit(reference[self.core_genes].to_numpy(np.float32),
                       reference[self.full_gene_order].to_numpy(np.float32))
        return self

    def predict(self, core_prediction):
        prediction = np.asarray(core_prediction, dtype=np.float32)
        if prediction.ndim != 2 or prediction.shape[1] != 785:
            raise ValueError('Expected [spots, 785] direct predictions.')
        return self.model.predict(prediction).astype(np.float32)

