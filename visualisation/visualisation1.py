import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import FakeData
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
# Use non-interactive Agg backend so we can save without error
import matplotlib
matplotlib.use('Agg')


# random dataset (8×8)
transform = transforms.Compose([transforms.ToTensor()])
dataset = FakeData(size=1000, image_size=(1, 8, 8), num_classes=10, transform=transform)
loader = DataLoader(dataset, batch_size=64, shuffle=True)


# one-hidden-layer ReLU net
class SimpleNet(nn.Module):
    def __init__(self, inp, hid, out):
        super().__init__()
        self.fc1 = nn.Linear(inp, hid)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hid, out)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        return self.fc2(x)


model = SimpleNet(input_dim := 8 * 8, hidden_dim := 32, output_dim := 10)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

# train until loss change < 1e-4
prev_loss = float('inf')
for epoch in range(200):
    total_loss = 0.0
    for X, y in loader:
        optimizer.zero_grad()
        loss = criterion(model(X), y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * X.size(0)
    avg_loss = total_loss / len(loader.dataset)
    if abs(prev_loss - avg_loss) < 1e-4:
        print(f"converged at epoch {epoch}, loss={avg_loss:.4f}")
        break
    prev_loss = avg_loss

# local-minimum parameters
base_params = torch.nn.utils.parameters_to_vector(model.parameters()).detach()

# two random, orthonormal directions
d1 = torch.randn_like(base_params)
d1 /= torch.norm(d1)
d2 = torch.randn_like(base_params)
d2 -= torch.dot(d2, d1) * d1
d2 /= torch.norm(d2)

# grid and loss there
alphas = np.linspace(-20, 20, 620)
betas = np.linspace(-20, 20, 620)
loss_vals = np.zeros((len(alphas), len(betas)))

X_fix, y_fix = next(iter(loader))
orig = base_params.clone()

for i, a in enumerate(alphas):
    for j, b in enumerate(betas):
        new_p = orig + a * d1 + b * d2
        torch.nn.utils.vector_to_parameters(new_p, model.parameters())
        loss_vals[i, j] = criterion(model(X_fix), y_fix).item()

# restore
torch.nn.utils.vector_to_parameters(orig, model.parameters())

# plot and save
A, B = np.meshgrid(alphas, betas)
L = loss_vals.T

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(A, B, L, rstride=1, cstride=1, edgecolor='none')
ax.set_xlabel('alpha')
ax.set_ylabel('beta')
ax.set_zlabel('loss')
plt.tight_layout()
plt.savefig('loss_landscape1.png', dpi=150, bbox_inches='tight')
print("Saved loss landscape to loss_landscape.png")
