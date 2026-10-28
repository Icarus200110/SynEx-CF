"""SynEx-CF: morphology, Fourier spatial context and Poincare fusion."""
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from src.contracts import LABEL2ID, BREAST_CLASSES

NUM_CLASSES = len(BREAST_CLASSES)


def grid_neighborhood(positions):
    """Return the ordered physical 3x3 slots; -1 marks missing spots."""
    if positions.ndim != 2 or positions.shape[1] != 2:
        raise ValueError('Expected [spots, 2] physical grid coordinates.')
    if not torch.allclose(positions.float(), positions.float().round()):
        raise ValueError('Use integer spot-grid coordinates, not pixel coordinates.')
    xy = positions.detach().round().long().cpu().tolist()
    lookup = {tuple(p): i for i, p in enumerate(xy)}
    if len(lookup) != len(xy):
        raise ValueError('Spot-grid coordinates must be unique within a section.')
    rows = [[lookup.get((x + dx, y + dy), -1)
             for dy in (-1, 0, 1) for dx in (-1, 0, 1)] for x, y in xy]
    return torch.tensor(rows, dtype=torch.long, device=positions.device)


class TissueTypeEmbedding(nn.Module):
    def __init__(self, embedding_dim=64, node_dim=128, num_classes=NUM_CLASSES):
        super().__init__()
        self.embedding = nn.Embedding(num_classes + 1, embedding_dim, padding_idx=0)
        self.projection = nn.Sequential(nn.Linear(9 * embedding_dim, node_dim),
                                        nn.LayerNorm(node_dim))
        self.num_classes = num_classes

    def forward(self, labels, neighborhood):
        if ((labels < -1) | (labels >= self.num_classes)).any():
            raise ValueError('Tissue indices must follow the classifier checkpoint ontology.')
        slots = labels[neighborhood.clamp_min(0)]
        slots = torch.where(neighborhood >= 0, slots, -torch.ones_like(slots))
        slots = slots[:, [4, 0, 1, 2, 3, 5, 6, 7, 8]]  # Center, then eight neighbors.
        # Unknown/missing -> 0; every known class is shifted by one.
        return self.projection(self.embedding(slots + 1).flatten(1))


class FourierKANLayer(nn.Module):
    """Learn an independent five-term Fourier function per input/output edge."""
    def __init__(self, in_features, out_features, grid_size=5):
        super().__init__()
        self.coefficients = nn.Parameter(torch.randn(2, out_features, in_features, grid_size)
                                         / (in_features * grid_size) ** 0.5)
        self.register_buffer('frequencies', torch.arange(1, grid_size + 1).float())

    def forward(self, x, source, target):
        phase = x.unsqueeze(-1) * self.frequencies
        transformed = (torch.einsum('nik,oik->no', phase.cos(), self.coefficients[0])
                       + torch.einsum('nik,oik->no', phase.sin(), self.coefficients[1]))
        out = transformed.new_zeros((len(x), transformed.shape[1]))
        return out.index_add(0, target, transformed[source])


class SpatialContextKAGNN(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, layers=2, graph_type='kan'):
        super().__init__()
        self.tissue = TissueTypeEmbedding(num_classes=num_classes)
        self.graph_type = graph_type
        layer = FourierKANLayer if graph_type == 'kan' else nn.Linear
        if graph_type not in ('kan', 'gcn'):
            raise ValueError('graph_type must be kan or gcn.')
        self.layers = nn.ModuleList([layer(128, 128) for _ in range(layers)])
        self.projection = nn.Linear(128, 128)

    def forward(self, labels, neighborhood):
        h = self.tissue(labels, neighborhood)
        active = neighborhood >= 0
        active[:, 4] = False  # Manuscript adjacency excludes self edges.
        target = torch.arange(len(h), device=h.device)[:, None].expand_as(neighborhood)[active]
        source = neighborhood[active]
        for layer in self.layers:
            if self.graph_type == 'kan':
                h = layer(h, source, target)
            else:
                # Standard symmetric-normalized GCN, including self loops.
                nodes = torch.arange(len(h), device=h.device)
                src, dst = torch.cat((source, nodes)), torch.cat((target, nodes))
                degree = torch.bincount(dst, minlength=len(h)).to(h.dtype)
                weights = (degree[src] * degree[dst]).rsqrt()
                messages = layer(h)[src] * weights[:, None]
                h = F.gelu(h.new_zeros(h.shape).index_add(0, dst, messages))
        return self.projection(h)


class MorphologyEncoder(nn.Module):
    """Official MambaVision pooled embedding; ResNet50 is an explicit ablation."""
    def __init__(self, backbone='mambavision', model_name='mamba_vision_T', pretrained=False):
        super().__init__()
        if backbone == 'mambavision':
            from mambavision import create_model
            from pathlib import Path
            cache = Path(torch.hub.get_dir()) / 'checkpoints'
            cache.mkdir(parents=True, exist_ok=True)
            # Load the original classification head before replacing it, so pretrained
            # ImageNet weights do not encounter a 1000-versus-256 shape mismatch.
            self.encoder = create_model(model_name, pretrained=pretrained,
                                        model_path=str(cache / f'{model_name}.pth.tar'))
            self.encoder.head = nn.Linear(self.encoder.head.in_features, 256)
        elif backbone == 'resnet50':
            from torchvision.models import resnet50, ResNet50_Weights
            self.encoder = resnet50(weights=ResNet50_Weights.DEFAULT if pretrained else None)
            self.encoder.fc = nn.Linear(self.encoder.fc.in_features, 256)
        else:
            raise ValueError(f'Unsupported morphology backbone: {backbone}')

    def forward(self, x):
        return self.encoder(x)


class SSDFNet(nn.Module):
    def __init__(self, dim=256, hyperedge_neighbors=5):
        super().__init__()
        if hyperedge_neighbors != 5:
            raise ValueError('The manuscript uses five other nearest nodes per hyperedge.')
        self.offset_net = nn.Sequential(nn.Conv2d(384, 128, 1), nn.GELU(),
                                        nn.Conv2d(128, 27, 1))
        self.kernel = nn.Parameter(torch.empty(9, 256, 256))
        nn.init.xavier_uniform_(self.kernel)
        self.tangent_projection = nn.Linear(384, dim, bias=False)
        self.semantic = nn.Sequential(nn.Linear(dim, dim, bias=False), nn.LayerNorm(dim))
        self.detail = nn.Linear(256, dim, bias=False)
        self.gate = nn.Linear(2 * dim, dim)
        self.register_buffer('kernel_grid',
                             torch.tensor([(dx, dy) for dy in (-1, 0, 1)
                                           for dx in (-1, 0, 1)], dtype=torch.float))

    @staticmethod
    def exp0(v):
        norm = v.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        return torch.tanh(norm).clamp_max(1 - 1e-5) * v / norm

    @staticmethod
    def log0(z):
        norm = z.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        return torch.atanh(norm.clamp_max(1 - 1e-5)) * z / norm

    @staticmethod
    def poincare_distance(x, y):
        x_norm, y_norm = x.square().sum(-1), y.square().sum(-1)
        squared = (x_norm[:, None] + y_norm[None, :] - 2 * (x @ y.T)).clamp_min(0)
        denominator = ((1 - x_norm)[:, None] * (1 - y_norm)[None, :]).clamp_min(1e-10)
        return torch.acosh((1 + 2 * squared / denominator).clamp_min(1 + 1e-7))

    def align_morphology(self, visual, spatial, positions):
        """Rasterize on physical coordinates, then sample offsets with bilinear interpolation."""
        xy = positions.round().long()
        xy = xy - xy.amin(0)
        width, height = int(xy[:, 0].max()) + 1, int(xy[:, 1].max()) + 1
        flat = xy[:, 1] * width + xy[:, 0]
        field = visual.new_zeros((height * width, 256)).index_copy(0, flat, visual)
        field = field.T.reshape(1, 256, height, width)
        joint = torch.cat((visual, spatial), dim=-1).T[None, :, :, None]
        parameters = self.offset_net(joint).squeeze(0).squeeze(-1).T
        offsets = parameters[:, :18].reshape(-1, 9, 2)
        modulation = parameters[:, 18:].sigmoid()
        sample_xy = xy[:, None, :].to(visual.dtype) + self.kernel_grid + offsets
        gx = 2 * sample_xy[..., 0] / max(width - 1, 1) - 1
        gy = 2 * sample_xy[..., 1] / max(height - 1, 1) - 1
        # grid_sample supports singleton raster axes through explicit zero padding.
        if width == 1 or height == 1:
            field = F.pad(field, (1, 1, 1, 1))
            gx = 2 * (sample_xy[..., 0] + 1) / (width + 1) - 1
            gy = 2 * (sample_xy[..., 1] + 1) / (height + 1) - 1
        grid = torch.stack((gx, gy), -1)[None]
        # Interpolation/coordinate arithmetic remains float32 for subpixel accuracy.
        with torch.autocast(device_type=visual.device.type, enabled=False):
            sampled = F.grid_sample(field.float(), grid.float(), mode='bilinear',
                                    padding_mode='zeros', align_corners=True)[0].permute(1, 2, 0)
        return torch.einsum('nkc,koc,nk->no', sampled, self.kernel, modulation)

    def hypergraph_semantics(self, joint):
        # Hyperbolic arithmetic is kept in float32 even under AMP.
        with torch.autocast(device_type=joint.device.type, enabled=False):
            z = self.exp0(self.tangent_projection(joint.float()))
            n = len(z)
            if n < 6:
                raise ValueError('A section needs at least six spots for center + five neighbors.')
            memberships = []
            # Topology is discrete; do not retain a quadratic autograd graph.
            with torch.no_grad():
                for start in range(0, n, 128):
                    distance = self.poincare_distance(z[start:start + 128], z)
                    rows = torch.arange(len(distance), device=z.device)
                    distance[rows, rows + start] = float('inf')
                    nearest = distance.topk(5, largest=False).indices
                    memberships.append(torch.cat(((rows + start)[:, None], nearest), dim=1))
            members = torch.cat(memberships)
            tangent = self.log0(z)
            edge_means = tangent[members].mean(1)
            # Every incident hyperedge contributes, including edges centered elsewhere.
            messages = edge_means[:, None, :].expand(-1, 6, -1).reshape(-1, z.shape[1])
            aggregated = torch.zeros_like(tangent).index_add(0, members.flatten(), messages)
            return self.semantic(aggregated)

    def forward(self, visual, spatial, positions):
        aligned = self.align_morphology(visual, spatial, positions)
        semantic = self.hypergraph_semantics(torch.cat((aligned, spatial), -1))
        detail = self.detail(aligned)
        alpha = self.gate(torch.cat((semantic, detail), -1)).sigmoid()
        return alpha * semantic + (1 - alpha) * detail


class SynEx_CF(nn.Module):
    def __init__(self, n_genes=785, semantic_classifier=None, num_classes=NUM_CLASSES,
                 backbone='mambavision', mamba_model='mamba_vision_T', pretrained=False,
                 graph_type='kan', fusion='ssdf', use_tissue=True, spot_chunk_size=128):
        super().__init__()
        if n_genes != 785:
            raise ValueError('Direct supervision in the manuscript uses exactly 785 genes.')
        if not 1 <= spot_chunk_size <= 128:
            raise ValueError('Morphology chunks must contain 1 to 128 spots.')
        if fusion not in ('ssdf', 'concat'):
            raise ValueError('fusion must be ssdf or concat.')
        self.use_tissue, self.fusion_type = use_tissue, fusion
        self.spot_chunk_size = spot_chunk_size
        self.semantic_classifier = semantic_classifier
        if semantic_classifier is not None:
            semantic_classifier.requires_grad_(False)
            semantic_classifier.eval()
        self.morphology = MorphologyEncoder(backbone, mamba_model, pretrained)
        self.spatial = SpatialContextKAGNN(num_classes=num_classes, graph_type=graph_type)
        self.fusion = SSDFNet() if fusion == 'ssdf' else nn.Identity()
        self.head = nn.Sequential(nn.Linear(256 if fusion == 'ssdf' else 384, 256),
                                  nn.LayerNorm(256), nn.GELU(), nn.Linear(256, n_genes))

    def train(self, mode=True):
        super().train(mode)
        if self.semantic_classifier is not None:
            self.semantic_classifier.eval()
        return self

    @torch.no_grad()
    def predict_tissue(self, patches):
        if self.semantic_classifier is None:
            raise ValueError('Load a classifier trained on this fold training sections.')
        return torch.cat([self.semantic_classifier(chunk[None])[0].argmax(-1)
                          for chunk in patches.split(self.spot_chunk_size)])

    def forward(self, patches, positions):
        """One entire section per call. Raw annotated tissue labels are never inputs."""
        batched = patches.ndim == 5
        if batched:
            if len(patches) != 1:
                raise ValueError('One complete tissue section is the unit of iteration.')
            patches, positions = patches[0], positions[0]
        if patches.shape[-3:] != (3, 224, 224):
            raise ValueError('Expected ImageNet-normalized 224x224 RGB patches.')
        neighborhood = grid_neighborhood(positions)
        labels = (self.predict_tissue(patches) if self.use_tissue
                  else torch.full((len(patches),), -1, device=patches.device, dtype=torch.long))
        features = []
        for chunk in patches.split(self.spot_chunk_size):
            if self.training and torch.is_grad_enabled():
                features.append(checkpoint(self.morphology, chunk, use_reentrant=False))
            else:
                features.append(self.morphology(chunk))
        visual = torch.cat(features)
        spatial = self.spatial(labels, neighborhood)
        fused = (self.fusion(visual, spatial, positions) if self.fusion_type == 'ssdf'
                 else torch.cat((visual, spatial), -1))
        output = self.head(fused)
        return output[None] if batched else output

