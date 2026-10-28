"""Run RACF on real normalized expression and explicit section provenance."""
import argparse
from pathlib import Path
import sys
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from RACF import RACF


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--database', required=True,
                        help='TSV with spot_id, section, then full log2-CPM expression.')
    parser.add_argument('--query', required=True, help='TSV with spot_id and 785 predicted core genes.')
    parser.add_argument('--core-genes', required=True, help='One gene per line, in query order.')
    parser.add_argument('--full-genes', required=True, help='One reference-covered gene per line.')
    parser.add_argument('--train-sections', nargs='+', required=True)
    parser.add_argument('--validation-sections', nargs='+', default=[])
    parser.add_argument('--test-sections', nargs='+', required=True)
    parser.add_argument('--clusters', type=int, default=10)
    parser.add_argument('--tau', type=float, default=0.1)
    parser.add_argument('--output', default='racf_full_predictions.tsv')
    args = parser.parse_args()
    model = RACF(args.database, database_gene_order=args.full_genes,
                 training_sections=args.train_sections,
                 validation_sections=args.validation_sections, test_sections=args.test_sections,
                 k=args.clusters, tau=args.tau)
    _, predictions = model.run(args.query, args.core_genes, calculate_final_expression=True)
    queries = pd.read_csv(args.query, sep='\t')
    output = pd.DataFrame(predictions, index=queries.iloc[:, 0], columns=model.full_gene_order)
    output.to_csv(args.output, sep='\t')


if __name__ == '__main__':
    main()

