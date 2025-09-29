# train_mnist_tiny.py
# PyTorch -> ONNX (fp32 + fp16) -> optional INT8 (static quantization)
# Requirements:
#   pip install torch torchvision onnx onnxconverter-common onnxruntime onnxruntime-tools

import os, io, json, math, random, time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import onnx
from onnxconverter_common import float16

# INT8 quant (optional)
try:
    from onnxruntime.quantization import (
        quantize_static, CalibrationDataReader, QuantFormat, QuantType
    )
    ORT_QUANT_AVAILABLE = True
except Exception:
    ORT_QUANT_AVAILABLE = False

# -----------------------
# Reproducibility
# -----------------------
SEED = 1337
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# -----------------------
# Tiny depthwise-separable CNN (very small file)
# -----------------------
class TinyDSCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        # 1x28x28 -> 8x28x28 (depthwise+pointwise) -> pool -> 16x14x14 -> pool -> 32x7x7 -> linear(10)
        self.dw1 = nn.Conv2d(1, 1, 3, padding=1, groups=1, bias=False)
        self.pw1 = nn.Conv2d(1, 8, 1, bias=False)
        self.dw2 = nn.Conv2d(8, 8, 3, padding=1, groups=8, bias=False)
        self.pw2 = nn.Conv2d(8, 16, 1, bias=False)
        self.pool = nn.MaxPool2d(2, 2)
        self.dw3 = nn.Conv2d(16, 16, 3, padding=1, groups=16, bias=False)
        self.pw3 = nn.Conv2d(16, 32, 1, bias=False)
        self.head = nn.Linear(32 * 7 * 7, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.pw1(self.dw1(x))))   # 8x14x14
        x = self.pool(F.relu(self.pw2(self.dw2(x))))   # 16x7x7
        x = F.relu(self.pw3(self.dw3(x)))              # 32x7x7
        x = x.reshape(x.size(0), -1)
        return self.head(x)

# -----------------------
# Data
# -----------------------
def get_loaders(batch_train=256, batch_test=512, num_workers=2):
    tfm = transforms.Compose([
        transforms.ToTensor(),                 # [0,1], 1x28x28
        transforms.Normalize(0.1307, 0.3081)  # MNIST stats
    ])
    train_ds = datasets.MNIST("./data", train=True, download=True, transform=tfm)
    test_ds  = datasets.MNIST("./data", train=False, download=True, transform=tfm)
    train_loader = DataLoader(train_ds, batch_size=batch_train, shuffle=True, num_workers=num_workers)
    test_loader  = DataLoader(test_ds,  batch_size=batch_test,  shuffle=False, num_workers=num_workers)
    return train_loader, test_loader

# -----------------------
# Train / Eval
# -----------------------
def train_and_eval(epochs=5, lr=2e-3, device="cpu", save_path="mnist_tiny.pt"):
    train_loader, test_loader = get_loaders()
    model = TinyDSCNN().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    best = 0.0

    for epoch in range(1, epochs + 1):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            opt.step()

        # quick eval
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x).argmax(1)
                correct += (pred == y).sum().item()
                total += y.size(0)
        acc = correct / total
        print(f"Epoch {epoch}: test acc {acc:.4f}")
        if acc > best:
            best = acc
            torch.save(model.state_dict(), save_path)
    print("Best accuracy:", best)
    return best

# -----------------------
# Export ONNX (FP32 + FP16)
# -----------------------
def export_onnx(fp32_path="mnist_tiny_fp32.onnx", fp16_path="mnist_tiny_fp16.onnx"):
    model = TinyDSCNN()
    model.load_state_dict(torch.load("mnist_tiny.pt", map_location="cpu"))
    model.eval()

    dummy = torch.zeros(1, 1, 28, 28, dtype=torch.float32)
    torch.onnx.export(
        model, dummy, fp32_path,
        input_names=["input"], output_names=["logits"],
        opset_version=13, do_constant_folding=True,
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}}
    )
    print("Exported FP32 ONNX:", fp32_path)

    m32 = onnx.load(fp32_path)
    m16 = float16.convert_float_to_float16(m32, keep_io_types=True)
    onnx.save(m16, fp16_path)
    print("Saved FP16 ONNX:", fp16_path)

# -----------------------
# INT8 Static Quantization (Optional)
# -----------------------
class MnistCalibReader(CalibrationDataReader):
    """
    Feeds a few hundred samples for static quantization calibration.
    Uses exporter input name 'input' and NCHW float32 batches.
    """
    def __init__(self, n_samples=200, batch=50):
        self.input_name = "input"
        tfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(0.1307, 0.3081)
        ])
        ds = datasets.MNIST("./data", train=True, download=True, transform=tfm)
        self.loader = iter(DataLoader(ds, batch_size=batch, shuffle=True))
        self.remaining = n_samples

    def get_next(self):
        if self.remaining <= 0:
            return None
        try:
            x, _ = next(self.loader)
        except StopIteration:
            return None
        self.remaining -= x.shape[0]
        return {self.input_name: x.numpy().astype("float32")}

def quantize_int8(fp32_path="mnist_tiny_fp32.onnx", int8_path="mnist_tiny_int8.onnx"):
    if not ORT_QUANT_AVAILABLE:
        print("onnxruntime.quantization not available — skipping INT8.")
        return
    calib = MnistCalibReader(n_samples=200, batch=50)
    quantize_static(
        model_input=fp32_path,
        model_output=int8_path,
        calibration_data_reader=calib,
        quant_format=QuantFormat.QOperator,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8
    )
    print("Saved INT8 ONNX:", int8_path)

# -----------------------
# Utility: show file sizes
# -----------------------
def show_sizes(paths):
    for p in paths:
        if Path(p).exists():
            kb = Path(p).stat().st_size / 1024
            print(f"{p}: {kb:.1f} KB")

# -----------------------
# Main
# -----------------------
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    # 1) Train and save best weights
    train_and_eval(epochs=5, lr=2e-3, device=device, save_path="mnist_tiny.pt")

    # 2) Export ONNX (FP32 + FP16)
    export_onnx("mnist_tiny_fp32.onnx", "mnist_tiny_fp16.onnx")

    # 3) Optional: INT8 static quantization (comment out if you don't want it)
    quantize_int8("mnist_tiny_fp32.onnx", "mnist_tiny_int8.onnx")

    # 4) Show resulting sizes
    show_sizes(["mnist_tiny_fp32.onnx", "mnist_tiny_fp16.onnx", "mnist_tiny_int8.onnx"])
