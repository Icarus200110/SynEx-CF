"""Single-GPU, section-level SynEx-CF training following the manuscript protocol."""
import argparse
import json
import random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from dataset import build_datasets
from SynEx_CF import SynEx_CF
from src.contracts import IMPLEMENTATION_VERSION
from src.semantic import train_classifier, load_classifier, classifier_spec
from src.evaluation import gene_wise_pcc, summarize_sections, bootstrap_sections_genes


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def section_names(value):
    return [s.strip() for s in value.split(',') if s.strip()] if value else None


@torch.no_grad()
def predict_sections(model, dataset, device):
    model.eval()
    predictions, scores = {}, {}
    for section in dataset:
        output = model(section['patches'].to(device), section['positions'].to(device))
        prediction = output.float().cpu().numpy()
        name = section['section']
        predictions[name] = prediction
        scores[name] = gene_wise_pcc(section['expression'].numpy(), prediction)
    return predictions, scores


def evaluate_outputs(model, datasets, splits, device, output_dir, args):
    predictions, direct_scores = predict_sections(model, datasets['test'], device)
    direct_summary = summarize_sections(direct_scores)
    direct_summary.update(bootstrap_sections_genes(direct_scores, seed=args.seed))
    summary = dict(direct=direct_summary, section_splits=splits,
                   implementation_version=IMPLEMENTATION_VERSION)
    pd.DataFrame(direct_scores, index=datasets['train'].gene_set).rename_axis('gene').to_csv(
        output_dir / 'direct_gene_pcc.tsv', sep='\t')
    full_scores = {}
    train = datasets['train']
    full_genes = train.full_gene_order
    if args.without_racf:
        from src.imputation import LinearImputer
        imputer = LinearImputer().fit(train)
    else:
        from RACF import RACF
        reference = train.reference_frame()
        retriever = RACF(reference, k=args.racf_clusters, n_candidates=args.racf_candidates,
                         n_precise=args.racf_top_k, tau=args.racf_tau,
                         database_gene_order=train.full_gene_order,
                         training_sections=splits['train'],
                         validation_sections=splits['validation'], test_sections=splits['test'],
                         seed=args.seed)
    for section in datasets['test']:
        name = section['section']
        prediction = predictions[name]
        if not args.without_racf:
            query = pd.DataFrame(prediction, columns=train.gene_set)
            query.insert(0, 'spot_id', section['spot_ids'])
            if retriever.faiss_index is None:
                _, full = retriever.run(query, train.gene_set, calculate_final_expression=True)
            else:
                retriever.spot_vectors = retriever._matrix(query, train.gene_set)
                _, full = retriever.generate_recommendations(calculate_final_expression=True)
        else:
            full = imputer.predict(prediction)
        truth = section['full_expression'].numpy()
        full_scores[name] = gene_wise_pcc(truth, full)
        frame = pd.DataFrame(full, index=section['spot_ids'], columns=full_genes)
        frame.index.name = 'spot_id'
        frame.to_csv(output_dir / f'{name}_predictions.tsv', sep='\t')
        np.savez_compressed(output_dir / f'{name}_direct.npz', prediction=prediction,
                            truth=section['expression'].numpy(),
                            spot_ids=np.asarray(section['spot_ids']), genes=np.asarray(train.gene_set))
    if full_scores:
        full_summary = summarize_sections(full_scores)
        full_summary.update(bootstrap_sections_genes(full_scores, seed=args.seed))
        summary['linear_full' if args.without_racf else 'racf_full'] = full_summary
        pd.DataFrame(full_scores, index=full_genes).rename_axis('gene').to_csv(
            output_dir / 'full_gene_pcc.tsv', sep='\t')
    (output_dir / 'metrics.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', choices=['her2st', 'cscc'], default='her2st')
    parser.add_argument('--data-root', default='data')
    parser.add_argument('--manifest', help='JSON section paths, including annotations where available.')
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--train-sections', '--train_samples', dest='train_sections')
    parser.add_argument('--validation-sections')
    parser.add_argument('--test-sections')
    parser.add_argument('--semantic-checkpoint')
    parser.add_argument('--semantic-model', choices=['spotvit_tiny', 'spotvit_small', 'spotvit_base'],
                        default='spotvit_small')
    parser.add_argument('--semantic-epochs', type=int, default=100)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--step-size', type=int, default=30)
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--clip-norm', type=float, default=1.0)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--output-dir', default='outputs/synex_cf')
    parser.add_argument('--backbone', choices=['mambavision', 'resnet50'], default='mambavision')
    parser.add_argument('--mamba-model', default='mamba_vision_T')
    parser.add_argument('--pretrained', action='store_true')
    parser.add_argument('--graph-type', choices=['kan', 'gcn'], default='kan')
    parser.add_argument('--fusion', choices=['ssdf', 'concat'], default='ssdf')
    parser.add_argument('--without-tissue', action='store_true')
    parser.add_argument('--without-racf', action='store_true',
                        help='Replace RACF with training-only linear full-expression imputation.')
    parser.add_argument('--experiment', default='main', choices=['main', 'ablation'])
    parser.add_argument('--run-name', default='full')
    parser.add_argument('--spot-chunk-size', type=int, default=128)
    parser.add_argument('--racf-clusters', type=int, default=10)
    parser.add_argument('--racf-candidates', type=int, default=5000)
    parser.add_argument('--racf-top-k', type=int, default=1000)
    parser.add_argument('--racf-tau', type=float, default=0.1)
    return parser


def main(args=None):
    args = build_parser().parse_args(args)
    if not 1 <= args.epochs <= 300 or not 1 <= args.semantic_epochs <= 300:
        raise ValueError('The manuscript protocol trains for at most 300 epochs.')
    if args.clip_norm <= 0:
        raise ValueError('Gradient clipping requires a positive norm.')
    if not 1 <= args.spot_chunk_size <= 128:
        raise ValueError('Use morphology chunks of 1 to 128 spots.')
    if not 1 <= args.racf_clusters <= args.racf_top_k <= args.racf_candidates or args.racf_tau <= 0:
        raise ValueError('RACF requires clusters <= top K <= candidates and positive temperature.')
    if Path(args.run_name).name != args.run_name or args.run_name in ('.', '..'):
        raise ValueError('run-name must be a single output directory name.')
    seed_everything(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    datasets, splits = build_datasets(
        args.data_root, args.data, args.fold, section_names(args.train_sections),
        section_names(args.validation_sections), section_names(args.test_sections),
        args.seed, args.manifest)
    train = datasets['train']
    output_dir = Path(args.output_dir) / args.experiment / args.run_name / f'fold_{args.fold}'
    if (output_dir / 'protocol.json').exists() or (output_dir / 'best.pth').exists():
        raise FileExistsError('Choose a new run-name/output-dir to preserve existing experiment results.')
    output_dir.mkdir(parents=True, exist_ok=True)
    semantic, semantic_config = None, None
    if not args.without_tissue:
        if args.semantic_checkpoint:
            semantic, saved = load_classifier(args.semantic_checkpoint, train.label_to_idx,
                                               splits['train'], device)
            semantic_config = saved['model_config']
        else:
            semantic = train_classifier(train, output_dir / 'tissue_classifier.pth', device,
                                        args.semantic_epochs, seed=args.seed,
                                        model_type=args.semantic_model,
                                        chunk_size=args.spot_chunk_size)
            semantic_config = classifier_spec(args.semantic_model)
    model_config = dict(n_genes=785, num_classes=max(1, len(train.label_to_idx)),
                        backbone=args.backbone, mamba_model=args.mamba_model,
                        pretrained=args.pretrained, graph_type=args.graph_type,
                        fusion=args.fusion, use_tissue=not args.without_tissue,
                        spot_chunk_size=args.spot_chunk_size)
    model = SynEx_CF(semantic_classifier=semantic, **model_config).to(device)
    optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    amp = device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda', enabled=amp)
    manifest = dict(section_splits=splits, core_genes=train.gene_set,
                    full_gene_order=train.full_gene_order, label_to_idx=train.label_to_idx,
                    model_config=model_config, semantic_model_config=semantic_config,
                    seed=args.seed, normalization='log2(1 + CPM); full library before core selection',
                    experiment=args.experiment, run_name=args.run_name,
                    dataset=args.data, implementation_version=IMPLEMENTATION_VERSION,
                    arguments=vars(args))
    (output_dir / 'protocol.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    best_pcc = -float('inf')
    for epoch in range(args.epochs):
        model.train()
        order = list(range(len(train)))
        random.Random(args.seed + epoch).shuffle(order)
        total_loss = 0.0
        for index in order:
            section = train[index]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=amp):
                # Chunk only morphology encoding; graph and hypergraph retain every section node.
                prediction = model(section['patches'].to(device), section['positions'].to(device))
                loss = nn.functional.mse_loss(prediction, section['expression'].to(device))
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm)
            scaler.step(optimizer)
            scaler.update()
            total_loss += float(loss.detach())
        _, validation_scores = predict_sections(model, datasets['validation'], device)
        validation_pcc = summarize_sections(validation_scores)['mean_pcc']
        if validation_pcc > best_pcc:
            best_pcc = validation_pcc
            torch.save(dict(**manifest, model_state_dict=model.state_dict(), epoch=epoch + 1,
                            validation_pcc=best_pcc), output_dir / 'best.pth')
        scheduler.step()
        print(f'Epoch {epoch + 1}: MSE={total_loss / len(train):.6f}; '
              f'validation gene-wise PCC={validation_pcc:.6f}')
    # Test predictions always use the checkpoint selected on independent validation sections.
    best = torch.load(output_dir / 'best.pth', map_location=device, weights_only=False)
    model.load_state_dict(best['model_state_dict'], strict=True)
    evaluate_outputs(model, datasets, splits, device, output_dir, args)


if __name__ == '__main__':
    main()

