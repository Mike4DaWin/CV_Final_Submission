# Attention Faithfulness and Adversarial-Patch Robustness in Plant Disease Classifiers

**CSCI 7090: Advanced Computer Vision & Deep Learning, Final Project (Fall 2026)**
Michael Ford

This project trains a baseline CNN and a fine-tuned ResNet-50 on PlantVillage (38 classes). It attacks both with universal adversarial patches and measures where each model "looks" with Grad-CAM. It then asks whether models that attend to the lesion are more robust to the patches. The checks are a PlantDoc cross-dataset test, a background-swap shortcut test, and IoU against 150 manually annotated lesion masks.

## Repository contents

| Path | What it is |
|---|---|
| `CV_Final_Pipeline_MichaelFord.ipynb` | The full pipeline, executed with saved outputs |
| `CV_Final_Ford_Presentation.pptx` | Final presentation slides |
| `outputs/` | Model checkpoints, training histories, cached predictions, CSV results and figures |
| `outputs/lesion_masks/` | 150 binary 224×224 lesion masks rasterized from the annotations |
| `annotation_images/` | The 150 annotated PlantVillage test images with their LabelMe `.json` polygons |

## Notebook structure

| Stage | Heading | Contents |
|---|---|---|
| 0 | Setup, data exploration, splits | Midterm pipeline: EDA, transforms, stratified train/val/test split, leakage checks |
| 1 | Model Training | Baseline CNN and ResNet-50 training, test-set evaluation, McNemar, bootstrap CIs, confusion matrices, error analysis |
| 2 | Universal Adversarial Patches | Patch optimization (5/10/15% of the image), attack success rate vs. a random-noise control, attractor classes |
| 3 | GradCAM + Attention Faithfulness | Grad-CAM, proxy leaf/lesion masks, leaf energy, lesion energy/lift, mask quality checks |
| 4 | Research Question | Per-image and per-class attention–robustness correlations, lesion-size confound, PlantDoc check, background shortcut test |
| 5 | Ground Truth | Manual-mask IoU, Spearman correlations with bootstrap CIs, cross-model comparison |

## Key results 

| | Baseline CNN | ResNet-50 |
|---|---|---|
| Test accuracy (95% bootstrap CI) | 88.98% [88.29, 89.63] | 99.71% [99.58, 99.82] |
| Macro-F1 | 0.859 | 0.996 |
| Patch ASR at 5 / 10 / 15% | 98.7 / 99.1 / 97.4% | 70.6 / 91.4 / 94.2% |
| PlantDoc (field images) accuracy | 7.1% | 18.0% |

McNemar: b = 885, c = 11, p = 2.7e-245. The notebook's "Results" and "Compare with Stage 4" sections give the full interpretation.

## Data

The datasets are not included in this repository because of their size. Download them separately:

- **PlantVillage**: https://data.mendeley.com/datasets/tywbtsjrjv/1 ("Plant leaf diseases dataset", Mendeley Data). The notebook uses the `Plant_leave_diseases_dataset_without_augmentation` folder (38 classes, 54,305 images) and skips `Background_without_leaves`.
- **PlantDoc**: https://github.com/pratikkayal/PlantDoc-Dataset

The setup cell reads the data relative to the notebook, so place it **next to** the repository folder:

```
parent_folder/
├── CV_Project_Data/
│   ├── PlantVillage/Plant_leave_diseases_dataset_without_augmentation/
│   └── PlantDoc/PlantDoc-Dataset-master/
└── <this repo>/
    └── CV_Final_Pipeline_MichaelFord.ipynb
```

On Google Colab, the notebook mounts Drive and reads from `MyDrive/CV_Project/` instead (see the "Setup & imports" cell).

## Running

```bash
pip install -r requirements.txt
```

Open the notebook and run the cells from top to bottom. The trained checkpoints and cached results in `outputs/` are loaded automatically, so nothing is retrained unless `FORCE_RECOMPUTE = True`. The original run was on an Apple-silicon Mac (MPS). The notebook falls back to CUDA or CPU automatically.

## AI-use disclosure

AI assistance (Claude) was used for debugging, proofreading and tutoring.
