import datetime
from torch import Tensor
import torch

DATE = datetime.datetime.now().strftime("%d-%m-%Y--%H-%M-%S")


def h_transform(x: Tensor) -> Tensor:
    e = 0.001
    return torch.sign(x) * (torch.sqrt(torch.abs(x) + 1) - 1) + e * x


def inverse_h_transform(x: Tensor) -> Tensor:
    e = 0.001
    return torch.sign(x) * (
        ((torch.sqrt(1 + 4 * e * (torch.abs(x) + 1 + e)) - 1) / (2 * e)) ** 2 - 1
    )


def transform_to_scalar(logits: Tensor, support_size: int) -> Tensor:
    probs = torch.softmax(logits, dim=1)
    support = (
        torch.tensor([x for x in range(-support_size, support_size + 1)])
        .expand(probs.shape)
        .float()
        .to(device=probs.device)
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
