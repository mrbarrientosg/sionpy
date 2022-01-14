from torch import Tensor
from torch.functional import F
import torch


def h_transform(x: Tensor) -> Tensor:
    e = 0.001
    sign = torch.sign(x).to(x.device)
    return sign * (torch.sqrt(torch.abs(x) + 1) - 1 + (x * e))


def inverse_h_transform(x: Tensor) -> Tensor:
    e = 0.001
    sign = torch.sign(x).to(x.device)
    return sign * (
        torch.pow((torch.sqrt(1 + e * 4 * (torch.abs(x) + 1 + e)) - 1) / (2 * e), 2) - 1
    )


def transform_to_scalar(logits: Tensor, support_size: int) -> Tensor:
    probs = F.softmax(logits, dim=1)
    support = (
        torch.tensor([x for x in range(-support_size, support_size + 1)])
        .float()
        .to(probs.device)
    )

    x = torch.sum(support * probs, dim=1, keepdim=True)

    return inverse_h_transform(x)


def transform_to_logits(x: Tensor, support_size: int) -> Tensor:
    x = h_transform(x)

    x = torch.clamp(x, -support_size, support_size)
    floor = x.floor()
    prob = x - floor
    logits = torch.zeros(x.shape[0], x.shape[1], 2 * support_size + 1).to(x.device)
    logits.scatter_(
        2, (floor + support_size).long().unsqueeze(-1), (1 - prob).unsqueeze(-1)
    )
    indexes = floor + support_size + 1
    prob = prob.masked_fill_(2 * support_size < indexes, 0.0)
    indexes = indexes.masked_fill_(2 * support_size < indexes, 0.0)
    logits.scatter_(2, indexes.long().unsqueeze(-1), prob.unsqueeze(-1))
    return logits
