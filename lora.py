import math
import torch
import torch.nn as nn

class LoRALayer(nn.Module):
    def __init__(self, original_layer, rank = 16, alpha=1.0):
        super(LoRALayer, self).__init__()
        self.original_layer = original_layer
        self.rank = rank
        self.alpha = alpha

        self.lora_A = nn.Parameter(torch.zeros((original_layer.out_features, rank)))
        self.lora_B = nn.Parameter(torch.zeros((rank, original_layer.in_features)))

        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        return self.original_layer.forward(x) + self.alpha * (self.lora_A @ self.lora_B @ x.T).T
    
class LoRAWrapper(nn.Module):
    def __init__(self, model: nn.Module, rank: int = 16, alpha: float = 1.0):
        super().__init__()
        self.model = model
        self.rank = rank
        self.alpha = alpha
        
        self.module_2_index = {}
        self.index_2_lora = nn.ModuleList()
        self._process_modules(model)
        print(f"self.module_2_index: {self.module_2_index}")
        print(f"self.index_2_lora: {self.index_2_lora}")
        
    def _process_modules(self, module: nn.Module):
        for name, child in module.named_children():
            if isinstance(child, nn.Linear):
                self.module_2_index[child] = len(self.index_2_lora)
                self.index_2_lora += [LoRALayer(child, self.rank, self.alpha)]
            else:
                self._process_modules(child)
        
    def forward(self, *args, **kwargs):
        hooks = []
        def forward_hook(module, input, output):
            if module in self.module_2_index:
                lora_layer = self.index_2_lora[self.module_2_index[module]]
                return lora_layer(input[0])
            return output
        for module in self.module_2_index:
            hooks.append(module.register_forward_hook(forward_hook))
        try:
            return self.model(*args, **kwargs)
        finally:
            for hook in hooks:
                hook.remove()
                
class LoraTestModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(3, 64)
        self.submodule = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            #nn.LayerNorm(64, eps=1.0e-6, elementwise_affine=True),
            nn.Linear(64, 64),
            nn.ReLU(),
            #nn.LayerNorm(64, eps=1.0e-6, elementwise_affine=True),
            nn.Linear(64, 3),
        )
    def forward(self, x):
        x = self.fc1(x)
        x = self.submodule(x)
        return x
    
def freeze_for_lora(lora_model):
    for name, param in lora_model.named_parameters():
        if "lora_" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
    
def unfreeze_for_lora(lora_model):
    for name, param in lora_model.named_parameters():
        param.requires_grad = True
    
def save_lora_parameters(model, filepath):
    lora_params = {}
    for name, module in model.named_modules():
        if isinstance(module, LoRALayer):
            lora_params[name] = {
                "lora_A": module.lora_A.detach().cpu().numpy(),
                "lora_B": module.lora_B.detach().cpu().numpy(),
                "alpha": module.alpha,
                "rank": module.rank,
            }
    torch.save(lora_params, filepath)
    print(f"LoRA parameters saved to {filepath}")
    
def load_lora_parameters(model, filepath):
    lora_params = torch.load(filepath)
    for name, module in model.named_modules():
        if name in lora_params and isinstance(module, LoRALayer):
            params = lora_params[name]
            module.lora_A.data = torch.tensor(params["lora_A"], device=module.lora_A.device)
            module.lora_B.data = torch.tensor(params["lora_B"], device=module.lora_B.device)
            module.alpha = params["alpha"]
            module.rank = params["rank"]
    print(f"LoRA parameters loaded from {filepath}")
    
def debug_weights(model):
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            weight_mean = module.weight.data.mean().item()
            print(f"Layer: {name}, Weight Mean: {weight_mean:.6f}")
    
import torch.optim as optim
import numpy as np
def generate_data(func, num_samples=10000):
    x = torch.rand((num_samples, 3)) * 2 * np.pi
    y = torch.tensor([func(*xi) for xi in x], dtype=torch.float32)
    return x, y

def test_model(model, train_data):
    model.eval()
    with torch.no_grad():
        x, y_true = train_data
        y_pred = model(x)
        mse = ((y_pred - y_true) ** 2).mean().item()
    print(f"Test MSE on Training Data: {mse:.4f}")
    return mse

from torch.utils.data import DataLoader, TensorDataset
def train_model(model, criterion, optimizer, data, epochs=100, batch_size=64):
    x, y = data
    dataset = TensorDataset(x, y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_x, batch_y in dataloader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {epoch_loss / len(dataloader):.4f}")

def funcA(x1, x2, x3):
    return np.sin(x1), np.cos(x2), np.sin(2 * x3) + np.cos(2 * x3)

def funcB(x1, x2, x3):
    return np.cos(x1), np.sin(x2), np.sin(x3) + np.cos(x3)

modelA = LoraTestModel()
modelB = LoRAWrapper(modelA, rank=4)

print("Training modelA...")
dataA = generate_data(funcA)
criterion = nn.MSELoss()
optimizerA = optim.Adam(modelA.parameters(), lr=0.001)

unfreeze_for_lora(modelA)
modelA.train()
train_model(modelA, criterion, optimizerA, dataA, epochs=20)
print("1-Testing modelA...")
test_model(modelA, dataA)
print("1-Weight modelA...")
debug_weights(modelA)
print("1-Weight modelB...")
debug_weights(modelB)
torch.save(modelA.state_dict(), "test_modelA.pth")
modelA.load_state_dict(torch.load("test_modelA.pth"))
print("2-Testing modelA...")
test_model(modelA, dataA)
print(f"modelA: {modelA}")
    

freeze_for_lora(modelB)
print("Training modelB...")
dataB = generate_data(funcB)
optimizerB = optim.Adam(modelB.parameters(), lr=0.001)
modelB.train()
freeze_for_lora(modelB)
train_model(modelB, criterion, optimizerB, dataB, epochs=20)
print(f"modelA: {modelA}")
print(f"modelB: {modelB}")
print("3-Testing modelA...")
test_model(modelA, dataA)
print("3-Testing modelB...")
test_model(modelB, dataB)
print("3-Weight modelA...")
debug_weights(modelA)
print("3-Weight modelB...")
debug_weights(modelB)
save_lora_parameters(modelB, "test_modelB_lora.pth")
load_lora_parameters(modelB, "test_modelB_lora.pth")
print("4-Testing modelA...")
test_model(modelA, dataA)
print("4-Testing modelB...")
test_model(modelB, dataB)
