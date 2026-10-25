import torch
import numpy as np

# -targets * log(p)
# log(p) = log(softmax) = log(numerator) - log(denominator)
# log(denominator) = log(sum exp)
# -target * (log(numerator) - log(denominator))
def cross_entropy(logits, targets, weights=None, ignore_index=None, reduction='mean'):
    if targets.dim() == 2:
        targets = targets.argmax(dim=1)
    targets = targets.long()
    logits_norm = logits - logits.max(dim=1, keepdim=True).values
    exp_logits_norm = torch.exp(logits_norm)
    log_sum_exp = torch.log(torch.sum(exp_logits_norm, dim=1)).unsqueeze(1)
    nll = -(logits_norm - log_sum_exp)
    if ignore_index is not None:
        mask = ignore_index != targets
        targets = targets * mask.to(targets.dtype)
        mask = mask.unsqueeze(1).to(nll.dtype)
        nll = nll * mask
    if weights is not None:
        nll = nll * weights
    nll = nll[torch.arange(len(targets), device=targets.device), targets]
    if reduction == 'mean':
        if weights is not None:
            weights_select = weights[targets]
            if ignore_index is not None:
                weights_select = mask.squeeze(1) * weights_select
            N = weights_select.sum()
        else:
            if ignore_index is not None:
                N = mask.sum()
            else:
                N = targets.shape[0]
        return nll.sum() / N
    else:
        return nll.sum() if reduction == 'sum' else nll


logits = np.array([[1000., 1000.], [1000., -1000.], [1000., -1000.]])
logits = torch.from_numpy(logits)
targets = torch.tensor([[1, 0], [1, 0], [0, 1]], dtype=torch.int32)
print(cross_entropy(logits, targets))


