
import numpy as np
import torch

x = torch.randn(1)

a = np.ones(5)
b = torch.from_numpy(a)
np.add(a, 1, out = a)
print(a)
print(b)

if torch.cuda.is_available():
    device = torch.device("cuda")
    y = torch.ones_like(x, device = device)
    x = x.to(device)
    z = x + y
    print(z)
    print(z.to("cpu", torch.double))
