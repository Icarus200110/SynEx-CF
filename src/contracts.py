"""Shared section-level split and tissue-label contracts."""
import random

BREAST_CLASSES = (
    'adipose tissue', 'breast glands', 'cancer in situ',
    'connective tissue', 'immune infiltrate', 'invasive cancer',
)
LABEL2ID = {name: i for i, name in enumerate(BREAST_CLASSES)}
IMPLEMENTATION_VERSION = 'manuscript-aligned-v2'


def assert_disjoint_sections(train, validation, test):
    groups = [list(train), list(validation), list(test)]
    if any(len(g) != len(set(g)) for g in groups):
        raise ValueError('Duplicate section identifiers in a split.')
    for i, left in enumerate(groups):
        for right in groups[i + 1:]:
            if set(left) & set(right):
                raise ValueError('Training, validation and test sections must be disjoint.')


def split_sections(names, fold=0, train_sections=None, validation_sections=None,
                   test_sections=None, validation_fraction=0.1, seed=42):
    names = sorted(set(names))
    if len(names) < 3:
        raise ValueError('At least three sections are required for train/validation/test.')
    if not 0 < validation_fraction < 1:
        raise ValueError('Validation fraction must lie strictly between zero and one.')
    if test_sections is None:
        if not 0 <= fold < len(names):
            raise ValueError('Fold index is outside the section list.')
        test_sections = [names[fold]]
    test_sections = list(test_sections)
    remaining = [s for s in names if s not in test_sections]
    if validation_sections is None:
        pool = list(train_sections) if train_sections is not None else remaining[:]
        random.Random(seed).shuffle(pool)
        n_val = max(1, round(len(pool) * validation_fraction))
        validation_sections = pool[:n_val]
        if train_sections is not None:
            train_sections = [s for s in train_sections if s not in validation_sections]
    validation_sections = list(validation_sections)
    if train_sections is None:
        train_sections = [s for s in remaining if s not in validation_sections]
    result = dict(train=list(train_sections), validation=validation_sections, test=test_sections)
    assert_disjoint_sections(**result)
    if any(not group for group in result.values()):
        raise ValueError('Each section split must be nonempty.')
    unknown = set().union(*map(set, result.values())) - set(names)
    if unknown:
        raise ValueError(f'Unknown sections: {sorted(unknown)}')
    return result
