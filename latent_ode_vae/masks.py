import torch

# abstract class for masking
class BaseMasker:
    @property
    def _dim(self) -> int:
        raise NotImplementedError("BaseMasker is an abstract class")
    
    def __call__(self, obs: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("BaseMasker is an abstract class")


class FirstN(BaseMasker):
    def __init__(self, n: int):
        self.n = n
    
    @property
    def _dim(self) -> int:
        return self.n

    def __call__(self, obs: torch.Tensor) -> torch.Tensor:
        assert obs.shape[-1] >= self.n, "obs.shape[-1] < self.n"
        return obs[..., :self.n]


class GaussianNoise(BaseMasker):
    def __init(self, std: float):
        self.std = std
    
    @property
    def _dim(self) -> int:
        return -1 # placeholder, dimension is the same as the input

    def __call__(self, obs: torch.Tensor) -> torch.Tensor:
        assert obs.shape[-1] > 0, "obs.shape[-1] <= 0"
        noise = torch.randn_like(obs) * self.std
        return obs + noise
