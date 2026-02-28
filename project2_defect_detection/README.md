# AI Quality Control — Synthetic Defect Detection

> **Mercedes-Benz Digitalisierung · KI Team · Interview Demo**

A production-structured ML pipeline that generates synthetic factory
part images, extracts statistical features, and trains a classifier
to detect five types of surface defects — mirroring the synthetic-data
approach used with NVIDIA Omniverse and Unity in real manufacturing AI.

---

## Demo Output

![Results](assets/defect_detection_results.png)

---

## Project Structure

```
project2_defect_detection/
│
├── main.py                  ← Entry point (run this)
├── requirements.txt
├── config/
│   └── config.ini           ← All parameters in one place
├── src/
│   ├── __init__.py
│   ├── generator.py         ← Synthetic image generation (5 classes)
│   ├── features.py          ← Statistical feature extraction (34 features)
│   ├── classifier.py        ← Random Forest training + evaluation
│   └── visualizer.py        ← Dashboard rendering
└── tests/
    └── test_pipeline.py     ← Unit tests (pytest)
```

---

## Quickstart

```bash
# 1. Clone
git clone https://github.com/YOUR_USERNAME/mercedes-ai-projects.git
cd mercedes-ai-projects/project2_defect_detection

# 2. Install
pip install -r requirements.txt

# 3. Run
python main.py
```

**Options:**
```bash
python main.py --n_per_class 400    # more training data
python main.py --no_show            # save only, no window
python main.py --help               # all options
```

**Run tests:**
```bash
pytest tests/ -v
```

---

## Defect Classes

| Class | Description | Visual signature |
|-------|-------------|-----------------|
| `normal` | Healthy part | Uniform mid-gray + soft highlight |
| `crack` | Surface crack | Thin dark line |
| `hole` | Impact / drill hole | Dark circle + bright metallic rim |
| `corrosion` | Rust / oxidation | Scattered dark spots |
| `scratch` | Tool scoring | Thin bright line |

---

## How It Works

### 1 · Synthetic Data Generation
Each image is 64×64 grayscale pixels (values 0–1).
Defects are rendered with physically motivated pixel patterns:
cracks as thin dark lines, holes as dark circles with bright rims, etc.
Camera noise is added to every image for realism.

**Why synthetic?**
Real factory defect images are rare and proprietary.
Synthetic generation gives unlimited labeled training data —
the same rationale behind NVIDIA Omniverse in production AI.

### 2 · Feature Extraction (34 features)
Raw pixels → 34 statistical descriptors per image:

| Group | Count | What it captures |
|-------|-------|-----------------|
| Global stats | 8 | Overall brightness distribution |
| Regional stats | 18 | 3×3 grid — where is the anomaly? |
| Edge gradients | 4 | Crack edges, hole rims |
| Texture | 4 | Dark/bright/mid ratios, entropy |

### 3 · Random Forest Classifier
100 decision trees, each trained on a random data + feature subset.
Final prediction = majority vote.
Feature importance shows which statistics matter most.

---

## Results

| Metric | Value |
|--------|-------|
| Overall accuracy | ~99.5% |
| Training time | < 5 s |
| Inference (1 image) | < 1 ms |
| Parameters | 0 (no neural network) |

---

## Roadmap

- [ ] CNN / ViT on raw pixels for comparison
- [ ] Grad-CAM visualizations (which pixels drove the decision?)
- [ ] Integration with NVIDIA Omniverse synthetic renderer
- [ ] ONNX export for edge deployment (camera unit)
- [ ] Streamlit live inspection demo

---

## Interview Notes

**"Why not a CNN?"**
> For 1000 images and 64×64 resolution, a CNN would overfit without
> heavy augmentation. Random Forest on handcrafted features is a
> strong, interpretable baseline. At production scale (millions of
> images), I would fine-tune a pretrained ResNet or ViT.

**"What is feature importance?"**
> It measures how much each feature reduces prediction error across
> all trees. It's a built-in explainability tool — I can tell the
> engineering team exactly which visual property (e.g. dark pixel
> ratio) drives crack detection.

**"How does synthetic data connect to Mercedes?"**
> Mercedes and NVIDIA use Omniverse to generate photorealistic
> synthetic training data for exactly this problem — avoiding the
> need to collect and label thousands of real defect images.
