import torch

print("built:", torch.backends.mps.is_built())
print("available:", torch.backends.mps.is_available())

if torch.backends.mps.is_available():
    x = torch.ones(1, device="mps")
    print(x)
else:
    print("MPS not available (macOS < 12.3 and/or no MPS-capable GPU).")
