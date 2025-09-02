import torch, os
sd = torch.load("models/deeplabv3.pth", map_location="cpu")
if isinstance(sd, torch.nn.Module):
    sd = sd.state_dict()
print("Total keys:", len(sd))
for i, k in enumerate(list(sd.keys())[:40]):
    print(i, k, getattr(sd[k], "shape", None))