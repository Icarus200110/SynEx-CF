"""Training and strict loading of fold-specific tissue classifiers."""
from pathlib import Path
import random
import torch
from torch import nn
from src.contracts import IMPLEMENTATION_VERSION, assert_disjoint_sections
from src.models.spotformer import create_spotresnet


def classifier_spec(model_type='spotvit_small'):
    return dict(model_type=model_type, img_size=224, patch_size=16)


def new_classifier(label_to_idx, model_config=None):
    if not label_to_idx or sorted(label_to_idx.values()) != list(range(len(label_to_idx))):
        raise ValueError('Classifier class IDs must be contiguous zero-based indices.')
    model_config = dict(model_config or classifier_spec())
    if model_config.get('img_size') != 224 or model_config.get('patch_size') != 16:
        raise ValueError('The tissue classifier must use 224-pixel images and 16-pixel tokens.')
    return create_spotresnet(num_classes=len(label_to_idx), **model_config)


def load_classifier(path, label_to_idx, allowed_sections=None, device='cpu'):
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    if checkpoint.get('implementation_version') != IMPLEMENTATION_VERSION:
        raise ValueError('This classifier predates the class-token/section-provenance implementation.')
    if checkpoint.get('label_to_idx') != label_to_idx:
        raise ValueError('Classifier ontology does not match this training fold.')
    provenance = checkpoint.get('section_splits', {})
    assert_disjoint_sections(provenance.get('train', []), provenance.get('validation', []),
                             provenance.get('test', []))
    used = set(provenance.get('train', [])) | set(provenance.get('validation', []))
    if not used:
        raise ValueError('Classifier checkpoint lacks training/selection section provenance.')
    if allowed_sections is not None and not used.issubset(set(allowed_sections)):
        raise ValueError('Classifier training or selection used sections outside the expression training fold.')
    model = new_classifier(label_to_idx, checkpoint['model_config'])
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    return model.to(device).eval(), checkpoint


def train_classifier(dataset, output_path, device, epochs=100, lr=5e-4, seed=42,
                     model_type='spotvit_small', chunk_size=128):
    """Split only the expression training sections for classifier fitting/selection."""
    if not dataset.label_to_idx:
        raise ValueError('Training annotations are required; provide a fold-specific annotated manifest.')
    if not 1 <= epochs <= 300 or not 1 <= chunk_size <= 128:
        raise ValueError('Use 1..300 epochs and chunks of at most 128 spots.')
    device = torch.device(device)
    annotated = [name for name in dataset.names if
                 any(str(x).strip().lower() in dataset.label_to_idx
                     for x in dataset.sections[name]['labels'])]
    if len(annotated) < 2:
        raise ValueError('At least two annotated training sections are needed for classifier selection.')
    indices = {name: dataset.names.index(name) for name in annotated}
    order = annotated[:]
    random.Random(seed).shuffle(order)
    n_val = max(1, round(0.1 * len(order)))
    splits = dict(train=order[n_val:], validation=order[:n_val], test=[])
    config = classifier_spec(model_type)
    model = new_classifier(dataset.label_to_idx, config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    amp = device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda', enabled=amp)
    best = -float('inf')
    output_path = Path(output_path)
    if output_path.exists():
        raise FileExistsError('Choose a new classifier output path to preserve the existing checkpoint.')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(epochs):
        model.train()
        sequence = splits['train'][:]
        random.Random(seed + epoch).shuffle(sequence)
        for name in sequence:
            section = dataset[indices[name]]
            labels = section['tissue_labels']
            valid = labels >= 0
            patches = section['patches'][valid]
            labels = labels[valid]
            if not len(labels):
                continue
            optimizer.zero_grad(set_to_none=True)
            for patch, target in zip(patches.split(chunk_size), labels.split(chunk_size)):
                patch, target = patch.to(device), target.to(device)
                with torch.autocast(device_type=device.type, enabled=amp):
                    logits = model(patch[None])[0]
                    loss = nn.functional.cross_entropy(logits, target, reduction='sum') / len(labels)
                scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for name in splits['validation']:
                section = dataset[indices[name]]
                for patch, target in zip(section['patches'].split(chunk_size),
                                         section['tissue_labels'].split(chunk_size)):
                    target = target.to(device)
                    with torch.autocast(device_type=device.type, enabled=amp):
                        output = model(patch.to(device)[None])[0]
                    valid = target >= 0
                    correct += int(((output.argmax(-1) == target) & valid).sum())
                    total += int(valid.sum())
        if not total:
            raise ValueError('No annotated validation spots for classifier selection.')
        accuracy = correct / total
        if accuracy > best:
            best = accuracy
            torch.save(dict(model_state_dict=model.state_dict(), model_config=config,
                            label_to_idx=dataset.label_to_idx,
                            idx_to_label={i: label for label, i in dataset.label_to_idx.items()},
                            section_splits=splits, validation_accuracy=best, epoch=epoch + 1,
                            implementation_version=IMPLEMENTATION_VERSION), output_path)
        scheduler.step()
    model, _ = load_classifier(output_path, dataset.label_to_idx, dataset.names, device)
    return model

