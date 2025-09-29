<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>MNIST Digit Classifier (ONNX Runtime Web) – README</title>
  <style>
    body {
      font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Inter, Helvetica, Arial;
      background: #0f1220;
      color: #eef1f7;
      margin: 0;
      padding: 20px;
      line-height: 1.6;
    }
    h1, h2 { margin-top: 1.2em; margin-bottom: 0.5em; }
    h1 { font-size: 1.8em; }
    h2 { font-size: 1.3em; }
    code, pre {
      background: #171a2b;
      color: #eaf0ff;
      padding: 2px 6px;
      border-radius: 6px;
    }
    pre { padding: 12px; overflow-x: auto; }
    ul { margin: 0 0 1em 1.2em; }
    a { color: #7c9cf5; text-decoration: none; }
    a:hover { text-decoration: underline; }
    hr { border: 0; height: 1px; background: rgba(255,255,255,0.1); margin: 2em 0; }
  </style>
</head>
<body>
  <h1>🖌️ MNIST Digit Classifier (ONNX Runtime Web)</h1>
  <p>An interactive browser demo where you can <strong>draw a digit (0–9)</strong> on a canvas and instantly get predictions using a <strong>tiny PyTorch CNN</strong> exported to <strong>ONNX</strong>. The model runs entirely <em>client‑side</em> with <a href="https://onnxruntime.ai/" target="_blank">onnxruntime‑web</a> (WebGPU/WASM) — no server required.</p>

  <hr/>
  <h2>✨ Features</h2>
  <ul>
    <li>🎨 <strong>Canvas Drawing</strong> – mouse & touch support with adjustable brush.</li>
    <li>🔍 <strong>Instant Prediction</strong> – 0–9 classification with probability bars.</li>
    <li>🖼️ <strong>28×28 Preview</strong> – see exactly what the model sees.</li>
    <li>⚡ <strong>Lightweight Models</strong> – <code>INT8</code> / <code>FP16</code> / <code>FP32</code> (all &lt; 1 MB).</li>
    <li>🌐 <strong>Static Hosting</strong> – works on GitHub Pages or any static host.</li>
  </ul>

  <hr/>
  <h2>📂 Project Structure</h2>
  <pre><code>.
├── index.html              # UI + inference (runs fully in-browser)
├── train.py                # PyTorch training & ONNX export script
├── mnist_tiny.pt           # Trained PyTorch weights
├── mnist_tiny_fp32.onnx    # Full precision ONNX
├── mnist_tiny_fp16.onnx    # Half precision ONNX
├── mnist_tiny_int8.onnx    # Quantized INT8 ONNX
└── README.html             # Project documentation
</code></pre>

  <hr/>
  <h2>🚀 How to Run (Frontend)</h2>
  <ol>
    <li>Place <code>index.html</code> and the <code>.onnx</code> models in the same folder.</li>
    <li>Open <code>index.html</code> in your browser, or publish via <strong>GitHub Pages</strong>.</li>
    <li>Draw a digit → click <strong>Predict</strong> → see the result!</li>
  </ol>
  <p><em>💡 On GitHub Pages, keep model paths relative. If you use a <code>models/</code> folder, update the paths in <code>index.html</code> accordingly.</em></p>

  <hr/>
  <h2>🛠️ Train & Export (PyTorch → ONNX)</h2>
  <ol>
    <li>Install requirements:
      <pre><code>pip install torch torchvision onnx onnxconverter-common onnxruntime onnxruntime-tools</code></pre>
    </li>
    <li>Run training & export:
      <pre><code>python train.py</code></pre>
      This will:
      <ul>
        <li>Train a tiny CNN on MNIST.</li>
        <li>Save PyTorch weights (<code>mnist_tiny.pt</code>).</li>
        <li>Export ONNX in FP32, FP16, and INT8 formats.</li>
      </ul>
    </li>
  </ol>

  <hr/>
  <h2>📸 Demo Preview</h2>
  <p><em>Add a GIF or screenshot of the demo in action here.</em></p>

  <hr/>
  <h2>🧩 Tech Stack</h2>
  <ul>
    <li><strong>PyTorch</strong> – training & export</li>
    <li><strong>ONNX</strong> – interoperable model format</li>
    <li><strong>onnxruntime‑web</strong> – browser inference (WebGPU/WASM)</li>
    <li><strong>Vanilla HTML/CSS/JS</strong> – clean interactive UI</li>
  </ul>

  <hr/>
  <h2>📈 Extensions</h2>
  <ul>
    <li>Replace MNIST with <strong>Fashion‑MNIST</strong> for clothing classification.</li>
    <li>Fine‑tune <strong>MobileNetV3‑Small</strong> for a small custom dataset (&lt;10 MB).</li>
    <li>Deploy multiple models and add a dropdown to switch between them.</li>
    <li>Improve UI with animations, themes, and probability charts.</li>
  </ul>

  <hr/>
  <h2>📜 License</h2>
  <p>MIT License – free to fork and modify.</p>

  <p><em>👩‍💻 Built as a portfolio demo to showcase in‑browser ML with ONNX Runtime Web.</em></p>
</body>
</html>
