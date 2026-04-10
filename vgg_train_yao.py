import torch, os
import random
import torch.nn as nn
import numpy as np
import brevitas.nn as qnn
from torch.utils.data import TensorDataset, DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

save_dir = r"C:\Users\Edward\Desktop\VGGtrain\RGB_image\dataset\output"
os.makedirs(save_dir, exist_ok=True)


#===== 實驗結果不亂飄，讓FINN好比較，可加可不加 =====
#seed = 42 #訓練隨機性固定
#random.seed(seed)
#np.random.seed(seed)
#torch.manual_seed(seed)
#torch.cuda.manual_seed_all(seed)
#torch.backends.cudnn.deterministic = True
#torch.backends.cudnn.benchmark = False


# ===== 載入資料 =====
X_train = np.load(r'C:\Users\Edward\Desktop\VGGtrain\RGB_image\dataset\output\X_train.npy')
X_val   = np.load(r'C:\Users\Edward\Desktop\VGGtrain\RGB_image\dataset\output\X_val.npy')
X_test  = np.load(r'C:\Users\Edward\Desktop\VGGtrain\RGB_image\dataset\output\X_test.npy')

y_train = np.load(r'C:\Users\Edward\Desktop\VGGtrain\RGB_image\dataset\output\y_train.npy')
y_val   = np.load(r'C:\Users\Edward\Desktop\VGGtrain\RGB_image\dataset\output\y_val.npy')
y_test  = np.load(r'C:\Users\Edward\Desktop\VGGtrain\RGB_image\dataset\output\y_test.npy')

# one-hot → label
y_train = np.argmax(y_train, axis=1)
y_val   = np.argmax(y_val, axis=1)
y_test  = np.argmax(y_test, axis=1)

# NHWC → NCHW
X_train = torch.tensor(X_train).permute(0,3,1,2).float()
X_val   = torch.tensor(X_val).permute(0,3,1,2).float()
X_test  = torch.tensor(X_test).permute(0,3,1,2).float()

y_train = torch.tensor(y_train)
y_val   = torch.tensor(y_val)
y_test  = torch.tensor(y_test)

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
test_loader  = DataLoader(TensorDataset(X_test, y_test), batch_size=32)

# ===== 模型定義 =====
class QuantCNN(nn.Module):
    def __init__(self, BIT):
        super().__init__()

        if BIT == 32:
            # FP32 baseline
            self.conv1 = nn.Conv2d(3, 4, 3)
            self.conv2 = nn.Conv2d(4, 8, 3)
            self.conv3 = nn.Conv2d(8, 12, 3)
            self.fc1 = nn.Linear(12*2*2, 16)
            self.fc2 = nn.Linear(16, 8)
            self.fc3 = nn.Linear(8, 4)
            self.relu = nn.ReLU()

        else:
            self.conv1 = qnn.QuantConv2d(3, 4, 3, weight_bit_width=BIT)
            self.conv2 = qnn.QuantConv2d(4, 8, 3, weight_bit_width=BIT)
            self.conv3 = qnn.QuantConv2d(8, 12, 3, weight_bit_width=BIT)

            self.fc1 = qnn.QuantLinear(12*2*2, 16, weight_bit_width=BIT)
            self.fc2 = qnn.QuantLinear(16, 8, weight_bit_width=BIT)
            self.fc3 = qnn.QuantLinear(8, 4, weight_bit_width=BIT)

            self.relu = qnn.QuantReLU(bit_width=BIT)

        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))

        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x
# ===== 訓練函數 =====
def train_model(BIT, epochs, lr):
    print(f"\n==== Training {BIT}-bit | epochs={epochs} ====")

    model = QuantCNN(BIT).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            out = model(xb)
            loss = loss_fn(out, yb)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch} Loss {total_loss:.4f}")

    # ===== 測試 =====
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for xb, yb in test_loader:
            pred = model(xb).argmax(dim=1)
            correct += (pred == yb).sum().item()
            total += yb.size(0)

    acc = correct / total
    print(f"{BIT}-bit Test Acc: {acc:.4f}")

    # ===== 存 ONNX =====
    dummy = torch.randn(1,3,32,32).to("cpu") # 轉ONNX需要CPU 
    model.to(device)
    torch.onnx.export(
        model,
        dummy,
        os.path.join(save_dir, f"model_{BIT}bit.onnx"),
        export_params=True,
        opset_version=11,
        input_names=['input'],
        output_names=['output']
    )

    return acc

# ===== 主實驗 =====
bit_config = {
    32: {"epoch": 15,  "lr": 1e-3},
    16: {"epoch": 20, "lr": 1e-3},
    8:  {"epoch": 25, "lr": 1e-3},
    4:  {"epoch": 40, "lr": 5e-4},
    2:  {"epoch": 60, "lr": 1e-4},
    1:  {"epoch": 80, "lr": 1e-4},
}
results = {}

for b, cfg in bit_config.items():
    acc = train_model(
        BIT=b,
        epochs=cfg["epoch"],
        lr=cfg["lr"]
    )
    results[b] = acc
print("\n==== Final Results ====")
for b in results:
    print(f"{b}-bit: {results[b]:.4f}")