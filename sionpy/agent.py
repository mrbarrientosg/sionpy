from typing import List
from torch import Tensor
from torch.distributions import Categorical

class ActorCriticAgent:
    def __call__(self, logprobs: Tensor) -> Tensor:
        probabilities = logprobs.exp().squeeze(dim=-1)
        distribution = Categorical(probs=probabilities)
        return distribution.sample().detach().cpu().numpy()
