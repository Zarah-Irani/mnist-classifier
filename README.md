<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>MNIST Digit Classifier (ONNX Runtime Web) – README</title>
  <style>
    :root{
      --bg:#0f1220; --card:#171a2b; --accent:#7c9cf5; --muted:#9aa3b2; --text:#eef1f7; --ok:#6ee7b7;
    }
    *{box-sizing:border-box}
    body{margin:0; font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Inter, "Helvetica Neue", Arial; color:var(--text); background:radial-gradient(1200px 600px at 10% -10%, #1b2145 0%, #0f1220 55%) fixed;}
    .wrap{max-width:980px; margin:40px auto; padding:0 20px; display:grid; gap:20px}
    .card{background:var(--card); border:1px solid rgba(124,156,245,.2); border-radius:16px; padding:22px; box-shadow:0 10px 30px rgba(0,0,0,.25)}
    h1,h2{margin:.2em 0 .4em}
    h1{font-size:28px}
    h2{font-size:20px}
    p{color:var(--text); line-height:1.6}
    .muted{color:var(--muted)}
    a{color:#9db4ff; text-decoration:none}
    a:hover{text-decoration:underline}
    ul{margin:0 0 0 1.25em}
    code, pre{background:#0c1126; color:#eaf0ff; border:1px solid rgba(124,156,245,.25); border-radius:10px}
    code{padding:2px 6px}
    pre{padding:14px; overflow:auto}
    .grid{display:grid; gap:14px}
    .badge{display:inline-flex; align-items:center; gap:8px; padding:6px 10px; border-radius:999px; background:#0e1326; border:1px solid rgba(124,156,245,.25); font-size:12px}
    .section{display:grid; gap:10px}
    .two{display:grid; grid-template-columns:1fr; gap:16px}
    @media (min-width:840px){ .two{grid-template-columns:1.1fr .9fr} }
    .kbd{font-variant-numeric: tabular-nums}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="card section">
      <span class="badge">README • HTML</span>
      <h1>🖌️ MNIST Digit Classifier (ONNX Runtime Web)</h1>
      <p>An interactive browser demo where you can <strong>draw a digit (0–9)</strong> on a canvas and instantly get predictions using a <strong>tiny PyTorch CNN</strong> exported to <strong>ONNX</strong>. The model runs entirely <em>client‑side</em> with <a href="https://onnxruntime.ai/" target="_blank" rel="noopener">onnxruntime‑web</a> (WebGPU/WASM) — no server required.</p>
    </div>

    <div class="two">
      <div class="card section">
        <h2>✨ Features</h2>
        <ul>
          <li>🎨 <strong>Canvas Drawing</strong> – mouse & touch support with adjustable brush.</li>
          <li>🔍 <strong>Instant Prediction</strong> – 0–9 with probability bars.</li>
          <li>🖼️ <strong>28×28 Preview</strong> – see exactly what the model sees.</li>
          <li>⚡ <strong>Lightweight Models</strong> – <code>INT8</code>/<code>FP16</code>/<code>FP32</code>; typically &lt; 1&nbsp;MB.</li>
          <li>🌐 <strong>Static Hosting</strong> – works on GitHub Pages or any static host.</li>
        </ul>
      </div>

      <div class="card section">
        <h2>🧩 Tech Stack</h2>
        <ul>
          <li><strong>PyTorch</strong> – training & export pipeline</li>
          <li><strong>ONNX</strong> – interoperable model format</li>
          <li><strong>onnxruntime‑web</strong> – browser inference (WebGPU/WASM)</li>
          <li><strong>Vanilla HTML/CSS/JS</strong> – clean, portable UI</li>
        </ul>
      </div>
    </div>

    <div class="card section">
      <h2>📂 Project Structure</h2>
      <pre><code>.
├── index.html              # UI + inference (runs fully in-browser)
├── train.py                # PyTorch training & ONNX export script
├── mnist_tiny.pt           # Trained PyTorch weights
├── mnist_tiny_fp32.onnx    # Full precision ONNX
├── mnist_tiny_fp16.onnx    # Half precision ONNX
├── mnist_tiny_int8.onnx    # Quantized INT8 ONNX
└── README.html             # This file (optional)
</code></pre>
    </div>

    <div class="two">
      <div class="card section">
        <h2>🚀 Run the Demo (Frontend)</h2>
        <ol>
          <li>Ensure <code>index.html</code> and the <code>.onnx</code> files sit in the same directory.</li>
          <li>Open <code>index.html</code> in your browser (or publish via GitHub Pages).</li>
          <li>Draw a digit → click <strong>Predict</strong> → view class & probabilities.</li>
        </ol>
        <p class="muted">Tip: On GitHub Pages, keep model paths relative (e.g., <code>mnist_tiny_fp16.onnx</code> or <code>models/mnist_tiny_fp16.onnx</code> if you move files into a <code>models/</code> folder).</p>
      </div>

      <div class="card section">
        <h2>🛠️ Train & Export (PyTorch → ONNX)</h2>
        <p>Install requirements:</p>
        <pre><code class="kbd">pip install torch torchvision onnx onnxconverter-common onnxruntime onnxruntime-tools</code></pre>
        <p>Run training/export:</p>
        <pre><code class="kbd">python train.py</code></pre>
        <p>This script trains a tiny CNN on MNIST, saves <code>.pt</code>, and exports <code>FP32</code>, <code>FP16</code>, and <code>INT8</code> ONNX models.</p>
      </div>
    </div>

    <div class="card section">
      <h2>📸 Demo Preview</h2>
      <p class="muted">(Optional) Add a GIF or screenshot of drawing & prediction in action.</p>
      <p><em>Example placeholder:</em></p>
      <pre><code>![demo-screenshot](https://user-images.githubusercontent.com/&lt;your-username&gt;/demo.png)</code></pre>
    </div>

    <div class="two">
      <div class="card section">
        <h2>📈 Extensions</h2>
        <ul>
          <li>Swap MNIST for <strong>Fashion‑MNIST</strong> (clothes).</li>
          <li>Fine‑tune <strong>MobileNetV3‑Small</strong> for a custom 3–4 class dataset (&lt;10&nbsp;MB ONNX).</li>
          <li>Host multiple models and add a dropdown to switch at runtime.</li>
          <li>Style upgrades: themes, animations, probability charts.</li>
        </ul>
      </div>
      <div class="card section">
        <h2>🔒 Notes</h2>
        <ul>
          <li>All inference is client‑side; no secrets or servers are needed.</li>
          <li>WebGPU is used when available; otherwise it falls back to WASM.</li>
          <li>Keep models under ~25&nbsp;MB for fast GitHub Pages loads (these are &lt;1&nbsp;MB).</li>
        </ul>
      </div>
    </div>

    <div class="card section">
      <h2>📜 License</h2>
      <p>MIT — free to fork and modify.</p>
      <p class="muted">Built as a portfolio demo to showcase in‑browser ML with ONNX Runtime Web.</p>
    </div>
  </div>
</body>
</html>
