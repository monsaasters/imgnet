
**A Universal Similarity Metric for Computer Vision**

[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.21232756-blue)](https://doi.org/10.5281/zenodo.21232756)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Author:** Imam Ghozali — Independent Researcher
📧 imam.gh98@gmail.com

---

## Overview

Traditional similarity metrics such as cosine similarity compare embedding vectors through **global angular relationships**.

**IMG** introduces a different paradigm: instead of comparing absolute vector values, IMG compares **local relational patterns** inside the embedding.

The proposed framework consists of three complementary metrics:

1. **IMG Sign Score**
2. **AMP IMG Score**
3. **Chain Score**

> **Note:** This work does **not** propose replacing cosine similarity. Instead, IMG is proposed as an *alternative* similarity metric. Experimental results suggest that the optimal metric depends on how the embedding itself is learned.

---

## Relational Learning Hypothesis

In Javanese, one expresses gratitude as **"matur suwun"**; in Sundanese, the same sentiment is conveyed as **"hatur nuhun"**. Despite different surface structures, both phrases encode identical meaning through internally consistent relational patterns.

This linguistic observation inspired the central hypothesis of this work:

> **Identity can be encoded through consistent relational patterns rather than absolute values.**

Instead of forcing embeddings to occupy a specific angular position, the proposed method trains the network to preserve **local relational consistency**.

Consequently, similarity is evaluated by comparing relational patterns rather than absolute vector orientation.

<img width="1907" height="848" alt="image" src="https://github.com/user-attachments/assets/dc9610ae-6666-45e8-b1ae-ada814d91d92" />

<img width="1907" height="981" alt="image" src="https://github.com/user-attachments/assets/16c28b28-a768-4665-b426-33fb00579856" />

---

## Installation

### Package

```bash
# Core only
pip install -e .

# With API extras
pip install -e .[api]

# With training extras
pip install -e .[train]

# Everything
pip install -e .[all]
```

### Conda

```bash
conda env create -f environment.yml
conda activate imgnet
```

---

## Usage Examples

### Example 1: Compare two embeddings

```python
import numpy as np
from imgnet.metrics import batch_compare

a = np.load("emb_a.npy")
b = np.load("emb_b.npy")
result = batch_compare(a, b)
print(result["vote"])  # MATCH / UNCERTAIN / DIFFERENT
```

### Example 2: Use a single metric

```python
from imgnet.metrics import img_sign_score, amp_img_score, chain_score, cosine_similarity

img = img_sign_score(a, b)
amp = amp_img_score(a, b)
chain = chain_score(a, b)
cos = cosine_similarity(a, b)
```

### Example 3: Load model and verify faces

```python
from imgnet.visualizer import compute_embedding, model_status
from imgnet.metrics import batch_compare
from PIL import Image
import numpy as np

print(model_status())

img1 = Image.open("photo1.jpg").convert("RGB")
img2 = Image.open("photo2.jpg").convert("RGB")

emb1 = compute_embedding(np.array(img1))
emb2 = compute_embedding(np.array(img2))
print(batch_compare(emb1, emb2))
```

### Example 4: Benchmark mode comparison

```bash
python benchmark_eval.py --input benchmark_realistic --format npy_dir
```

### Example 5: Threshold sweep

```bash
python threshold_sweep.py --input benchmark_realistic --format npy_dir
```

### Example 6: Robustness benchmark

```bash
python robustness_by_mode.py --input benchmark_realistic --format npy_dir
```

### Example 7: Generate plots

```bash
python plot_benchmarks.py
# Outputs PNG files to benchmark_plots/
```

### Example 8: Train a model

```python
from imgnet.train import IMGNet, TrainConfig, build_trainer, fit

cfg = TrainConfig(
    data_root="./data/casia-webface",
    ckpt_root="./checkpoints",
    num_epochs=50,
    hybrid_mode=True,
    hybrid_cosine_weight=0.5,
)
trainer = build_trainer(cfg)
model = fit(trainer)
```

---

## Troubleshooting

### `ModuleNotFoundError: No module named 'torch'`

The `imgnet` core package does not require PyTorch. If you see this error,
you are importing a training-related module. Install torch explicitly:

```bash
pip install torch torchvision
```

### `model_loaded: false` in web demo

The demo falls back to a deterministic embedding when:
- PyTorch is not installed
- MTCNN is not installed
- Checkpoint `best_model_epoch39_plateau.pth` is missing

To use the real IMGNet model:
1. Install torch + torchvision + facenet-pytorch
2. Place the checkpoint in one of these paths:
   - `best_model_epoch39_plateau.pth` in repo root
   - `~/.imgnet/best_model_epoch39_plateau.pth`
   - Or keep the existing Windows path if available

### Editable install fails with `setuptools.backends`

Use a modern setuptools version:

```bash
pip install --upgrade pip setuptools wheel
pip install -e .
```

### Tests fail with `batch_norm` error

Some PyTorch versions require batch size > 1 during training mode.
The tests now use `.eval()` and batch size 2 to avoid this.

---

## Voting System
 
Three metrics, one threshold (from IMG Sign sweep):
 
```
2/3 or 3/3 pass → ✅ MATCH
1/3 pass        → ⚠️  UNCERTAIN
0/3 pass        → ❌ DIFFERENT
```
---
Experimental Findings (Preliminary)

<img width="1551" height="606" alt="image" src="https://github.com/user-attachments/assets/1c95c525-cdc9-4eb7-8aca-8327ea863b7d" />
same person


<img width="1540" height="624" alt="image" src="https://github.com/user-attachments/assets/878663d7-f762-4faa-8349-8793e889f197" />
diff person

During development of the interactive ablation visualizer, a preliminary observation was made using the custom polygon occlusion tool:

Observation: Region-specific embedding sensitivity

When occluding specific facial regions (e.g., right eye) using a custom polygon mask and comparing the resulting embedding changes across two different individuals:


Same person, different photos: occluding the same region produces delta spikes at similar embedding dimensions across both photos
Different people: occluding the same region produces delta spikes at different embedding dimensions, or in some cases near-zero delta for one person (e.g., when glasses obscure the region)

Example (custom polygon, right eye region):

Same person:
  Photo 1: changed 4/1014 windows = 0.4%  spike_delta: 110
  Photo 2: changed 1/1014 windows = 0.1%  spike_delta: 100
  → spike locations visually correlated

Different people:
  Photo 1: changed 4/1014 windows = 0.4%  spike_delta: 110
  Photo 2: changed 0/1014 windows = 0.0%  spike_delta: 97
  → spike locations differ significantly

This observation suggests that the IMG Sign MSE loss function, through its overlapping sliding window structure, may induce implicit spatial organization in the embedding space — where different facial regions influence different embedding dimensions. However, this has not yet been formally tested and should be treated as a preliminary observation pending rigorous evaluation.


⚠️ This is an informal observation from the visualization tool, not a validated experimental result. Formal ablation study with multiple identities and statistical analysis is planned as future work.
---
## Domain-Agnostic Potential (Beyond Computer Vision)

While evaluated on face verification, the core mathematics of the **IMG Framework** are inherently domain-agnostic. Because it discards absolute magnitude dependency and focuses entirely on local sign-pattern agreements, this framework can be generalized to non-visual embeddings:

*   **Audio & Speech Processing:** By applying IMG to audio spectrogram embeddings, the metric can eliminate amplitude/volume variations (gain changes), establishing a noise-robust framework for voice biometrics.
*   **Structural Bioinformatics:** In protein structural analysis, exact physical distances fluctuate due to environment/simulations. IMG can be applied to capture invariant relational topology patterns between amino acids rather than relying on strict absolute spatial coordinates.

---
## ONNX Export

IMGNet Conv10 is available in ONNX format for deployment without PyTorch.

**Why it was non-trivial to export:**
SW Block uses non-standard operations (boolean mask indexing + reflect padding) that are not directly ONNX-compatible. The solution: boolean mask replaced with explicit neighbor loops, and reflect padding emulated via `flip + concat` — mathematically identical but fully ONNX-safe across all runtimes.

**Verification:**
```
PyTorch vs ONNX cosine similarity : 1.000000
PyTorch vs ONNX max difference    : 0.00000033
```

**Inference:**
```python
import onnxruntime as rt
import numpy as np

sess     = rt.InferenceSession("imgnet_conv10_epoch39.onnx")
inp_name = sess.get_inputs()[0].name

# img: np.uint8 (112, 112, 3) — face aligned to 112×112
t   = img.astype(np.float32) / 255.0
t   = t.transpose(2, 0, 1)[np.newaxis]    # (1, 3, 112, 112)
emb = sess.run(None, {inp_name: t})[0][0]  # (1024,)
```

**Threshold** (from LFW benchmark sweep, epoch 39): `0.79`

```python
n_pass = sum([img_sign(e1,e2) >= 0.79,
              amp_img(e1,e2)  >= 0.79,
              chain(e1,e2)    >= 0.79])
# 2/3 or 3/3 → MATCH | 1/3 → UNCERTAIN | 0/3 → DIFFERENT
```
---

## Conclusion

IMG is proposed as an alternative similarity metric rather than a replacement for cosine similarity. Experiments indicate that cosine similarity performs best for embeddings trained with angular-margin objectives, while IMG Sign performs best for embeddings trained with the proposed relational objective. The framework is model-agnostic and can be applied to embeddings generated by different architectures.
 
---
 
## Occlusion Robustness Benchmark
 
To demonstrate that IMG metrics can be more stable than cosine under
perturbations, this repo includes a lightweight benchmark that applies
synthetic noise/occlusion directly to embedding vectors and compares metric
degradation.
 
Run:
```bash
python -m imgnet.benchmark
```
 
Or use it programmatically:
```python
from imgnet.benchmark import run_benchmark, summarize, to_csv, BenchmarkConfig, EmbeddingPair
import numpy as np
 
pairs = [
    EmbeddingPair("same_1", "same", np.random.randn(1024), np.random.randn(1024)),
    EmbeddingPair("diff_1", "different", np.random.randn(1024), np.random.randn(1024)),
]
 
results = run_benchmark(pairs, BenchmarkConfig())
print(summarize(results))
to_csv(results, Path("benchmark_results.csv"))
```
 
Outputs:
- `benchmark_results.csv`
- `benchmark_results.json`
 
---
 
## Training

This repo includes a modular training pipeline for IMGNet-style models.

Key features:
- Reproduced `SWBlock` + `Conv2-10` architecture
- Relational loss based on IMG Sign Score MSE
- `PairDataset` with positive/negative pairs and augmentation
- Checkpointing with resume support
- `TrainConfig` dataclass for reproducible experiments
- Optional **hybrid mode**: combine IMG Sign loss with cosine loss

Example:
```python
from imgnet.train import IMGNet, TrainConfig, build_trainer, fit

cfg = TrainConfig(
    data_root="./data/casia-webface",
    ckpt_root="./checkpoints",
    num_epochs=50,
    hybrid_mode=True,
    hybrid_cosine_weight=0.5,
)
trainer = build_trainer(cfg)
model = fit(trainer)
```

---
 
## Liveness & Reporting
 
The demo app also includes a lightweight liveness-check prototype and report
export endpoints. These are not a full anti-spoofing model, but demonstrate
how IMG metrics could be used in a liveness workflow.
 
Web demo endpoints:
- `POST /api/liveness` — heuristic liveness check from 2 uploaded images
- `POST /api/report/csv` — export comparison/liveness result as CSV
- `POST /api/report/json` — export result as JSON with generated timestamp
 
UI:
- **Check Liveness** button on the demo page
- **Export CSV** / **Export JSON** buttons to download result files
 
---
 
## Offline KYC App
 
This fork also includes an offline KYC verification app under `kyc_app/`.
It uses the same IMGNet metrics and model inference path, but wrapped
in a verification flow:
 
1. Upload selfie + KTP photo
2. Detect/crop faces using MTCNN
3. Generate embeddings using IMGNet model
4. Compare with IMG metrics + voting
5. Return vote + confidence
6. Export CSV/JSON report
 
Run:
```bash
PYTHONPATH=src uvicorn kyc_app.main:app --host 0.0.0.0 --port 8012
# Open http://0.0.0.0:8012
```
 
Endpoints:
- `GET /` — KYC UI
- `GET /api/health` — backend + model status
- `POST /api/kyc/verify` — verify selfie vs KTP
- `POST /api/kyc/report/csv` — export result as CSV
- `POST /api/kyc/report/json` — export result as JSON
 
---
## Citation
 
If you use this work, please cite via:
- **Zenodo (DOI):** https://doi.org/10.5281/zenodo.21232755
- **GitHub:** https://github.com/imamgh11/imgnet
- **Hugging Face:** https://huggingface.co/imghost11/imgnetV1

---
 
## License
 
This project is licensed under the **MIT License**.
