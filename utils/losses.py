import torch
import torch.nn as nn
import torch.nn.functional as F


def pairwise_cosine_distance(x, y):
    """
    Args:
        x: Tensor with shape (N, D).
        y: Tensor with shape (M, D).

    Returns:
        Tensor with shape (N, M), where each entry is 1 - cosine_similarity.
    """
    x = F.normalize(x, p=2, dim=1)
    y = F.normalize(y, p=2, dim=1)
    return 1 - torch.mm(x, y.t())


class CountNormalizedCosineCompactnessLoss(nn.Module):
    """
    NC1 compactness loss using cosine distance to class means, normalized by n_k.
    """
    count_power = 1.0

    def __init__(self, num_classes=10, feat_dim=128, device='cuda:0',
                 occurrence_list=None, fixed_means=False):
        super().__init__()
        if occurrence_list is None:
            raise ValueError('occurrence_list is required for class-count normalization.')

        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.fixed_means = fixed_means

        means = torch.randn(num_classes, feat_dim, device=device)
        if fixed_means:
            print('#####################Fixed means##################')
            self.means = means
        else:
            self.means = nn.Parameter(means)

        counts = torch.as_tensor(occurrence_list, dtype=torch.float32, device=device)
        self.class_counts = counts

    def forward(self, features, labels):
        batch_size = features.size(0)
        distances = pairwise_cosine_distance(features, self.means)

        classes = torch.arange(self.num_classes, device=features.device).long()
        label_mask = labels.unsqueeze(1).eq(classes.expand(batch_size, self.num_classes))
        seen_class_mask = self.class_counts.unsqueeze(0) > 0

        class_distances = distances * label_mask.float() * seen_class_mask.float()
        distance_sum = torch.sum(class_distances, dim=0)
        normalizer = self.class_counts.pow(self.count_power) + 1e-10

        loss = (distance_sum / normalizer).clamp(min=1e-12, max=1e+12).sum()
        return loss / self.num_classes, self.means


class SqrtCountNormalizedCosineCompactnessLoss(CountNormalizedCosineCompactnessLoss):
    """
    NC1 compactness loss using cosine distance to class means, normalized by sqrt(n_k).
    """
    count_power = 0.5

    def __init__(self, *args, **kwargs):
        print('Use SqrtCountNormalizedCosineCompactnessLoss, which normalizes by sqrt(n_k).')
        super().__init__(*args, **kwargs)


class CenteredMeanAngleSeparationLoss(nn.Module):
    """
    NC2 separation loss: maximize each centered class mean's minimum angle to others.
    """
    def __init__(self):
        super().__init__()

    def forward(self, means):
        global_mean = means.mean(dim=0)
        centered_means = means - global_mean
        normalized_means = F.normalize(centered_means, p=2, dim=1)
        cosine = torch.matmul(normalized_means, normalized_means.t())

        cosine = cosine - 2. * torch.diag(torch.diag(cosine))
        max_cosine = cosine.max().clamp(-0.99999, 0.99999)
        nearest_cosine = cosine.max(dim=1)[0].clamp(-0.99999, 0.99999)
        loss = -torch.acos(nearest_cosine).mean()

        return loss, max_cosine


COMPACTNESS_LOSSES = {
    'CountNormalizedCosineCompactnessLoss': CountNormalizedCosineCompactnessLoss,
    'SqrtCountNormalizedCosineCompactnessLoss': SqrtCountNormalizedCosineCompactnessLoss,
    # Backward-compatible config names; the old classes themselves were removed.
    'NC1Loss_v2_cosine': CountNormalizedCosineCompactnessLoss,
    'NC1Loss_v5_cosine': SqrtCountNormalizedCosineCompactnessLoss,
}

SEPARATION_LOSSES = {
    'CenteredMeanAngleSeparationLoss': CenteredMeanAngleSeparationLoss,
    # Backward-compatible config name; the old class itself was removed.
    'NC2Loss': CenteredMeanAngleSeparationLoss,
}


def _build_loss(registry, name, *args, **kwargs):
    try:
        loss_cls = registry[name]
    except KeyError as exc:
        available = ', '.join(sorted(registry))
        raise ValueError(f'Unknown loss "{name}". Available losses: {available}') from exc
    return loss_cls(*args, **kwargs)


def _build_supervised_loss(name):
    try:
        loss_cls = getattr(nn, name)
    except AttributeError as exc:
        raise ValueError(f'Unknown supervised loss "{name}" in torch.nn.') from exc
    return loss_cls()


class NeuralCollapseLoss(nn.Module):
    def __init__(self, sup_criterion, lambda1, lambda2, lambda_CE=1.0,
                 nc1='CountNormalizedCosineCompactnessLoss',
                 nc2='CenteredMeanAngleSeparationLoss',
                 num_classes=1920, feat_dim=2000, device='cuda:0',
                 occurrence_list=None, fixed_means=False):
        super().__init__()
        self.compactness_loss = _build_loss(
            COMPACTNESS_LOSSES,
            nc1,
            num_classes,
            feat_dim,
            device,
            occurrence_list,
            fixed_means,
        )
        self.separation_loss = _build_loss(SEPARATION_LOSSES, nc2)
        self.sup_criterion = _build_supervised_loss(sup_criterion)
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda_CE = lambda_CE
        self.num_classes = num_classes
        self.feat_dim = feat_dim

    @property
    def means(self):
        return self.compactness_loss.means

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        legacy_prefix = prefix + 'NC1.'
        new_prefix = prefix + 'compactness_loss.'
        for key in list(state_dict.keys()):
            if key.startswith(legacy_prefix):
                state_dict[new_prefix + key[len(legacy_prefix):]] = state_dict.pop(key)
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def forward(self, y_pred, labels, features):
        sup_loss = self.sup_criterion(y_pred, labels)
        compactness_loss, means = self.compactness_loss(features, labels)
        separation_loss, max_cosine = self.separation_loss(means)

        loss = (
            self.lambda_CE * sup_loss
            + self.lambda1 * compactness_loss
            + self.lambda2 * separation_loss
        )
        return loss, (sup_loss, compactness_loss, separation_loss, max_cosine, means)

    def set_lambda(self, lambda1, lambda2):
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        print(f'Set weights: lambda1={self.lambda1}, lambda2={self.lambda2}, lambda_CE={self.lambda_CE}')

    def set_lambda_CE(self, lambda_CE):
        self.lambda_CE = lambda_CE
        print(f'Set weights: lambda1={self.lambda1}, lambda2={self.lambda2}, lambda_CE={self.lambda_CE}')

    def freeze_means(self):
        if self.compactness_loss.fixed_means:
            print('The class means are fixed; no need to freeze them.')
            return
        self.compactness_loss.means.requires_grad = False
        print('Freeze the class means.')

    def unfreeze_means(self):
        if self.compactness_loss.fixed_means:
            print('The class means are fixed; no need to unfreeze them.')
            return
        self.compactness_loss.means.requires_grad = True
        print('Unfreeze the class means.')
