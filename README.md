# 🖌️ MNIST Digit Classifier (ONNX Runtime Web)

An interactive browser demo where you can **draw a digit (0–9)** on a canvas and instantly get predictions using a **tiny PyTorch CNN** exported to **ONNX**. The model runs entirely *client-side* with [onnxruntime-web](https://onnxruntime.ai/) (WebGPU/WASM) — no server required.

🌐 **Live Demo:** [https://zarah-irani.github.io/mnist-classifier/](https://zarah-irani.github.io/mnist-classifier/)

---

## ✨ Features
- 🎨 **Canvas Drawing** – mouse & touch support with adjustable brush.
- 🔍 **Instant Prediction** – 0–9 classification with probability bars.
- 🖼️ **28×28 Preview** – see exactly what the model sees.
- ⚡ **Lightweight Models** – `INT8` / `FP16` / `FP32` (all < 1 MB).
- 🌐 **Static Hosting** – works on GitHub Pages or any static host.

---

## 📂 Project Structure
```
.
├── index.html              # UI + inference (runs fully in-browser)
├── train.py                # PyTorch training & ONNX export script
├── mnist_tiny.pt           # Trained PyTorch weights
├── mnist_tiny_fp32.onnx    # Full precision ONNX
├── mnist_tiny_fp16.onnx    # Half precision ONNX
├── mnist_tiny_int8.onnx    # Quantized INT8 ONNX
└── README.md               # This README (Markdown-safe)
```

---

## 🚀 Run the Demo (Frontend)
1. Place `index.html` and the `.onnx` models in the same folder.
2. Open `index.html` in your browser, or publish via **GitHub Pages**.
3. Draw a digit → click **Predict** → see the result!

> **Tip:** On GitHub Pages, keep model paths relative. If you use a `models/` folder, update the paths in `index.html` accordingly.

---

## 🛠️ Train & Export (PyTorch → ONNX)
1. Install requirements:
   ```bash
   pip install torch torchvision onnx onnxconverter-common onnxruntime onnxruntime-tools
   ```
2. Train & export:
   ```bash
   python train.py
   ```
   This will:
   - Train a tiny CNN on MNIST.
   - Save PyTorch weights (`mnist_tiny.pt`).
   - Export ONNX in FP32, FP16, and INT8 formats.

---

## 📸 Demo Preview
_Add a GIF or screenshot of the demo in action here._

---

## 🧩 Tech Stack
- **PyTorch** – training & export
- **ONNX** – interoperable model format
- **onnxruntime-web** – browser inference (WebGPU/WASM)
- **Vanilla HTML/CSS/JS** – clean interactive UI

---

## 📈 Extensions
- Replace MNIST with **Fashion-MNIST**.
- Fine-tune **MobileNetV3-Small** for a small custom dataset (<10 MB).
- Deploy multiple models and add a dropdown to switch between them.
- Improve UI with animations, themes, and probability charts.

---

## 📜 License
MIT — free to fork and modify.

> 👩‍💻 Built as a portfolio demo to showcase in-browser ML with ONNX Runtime Web.
