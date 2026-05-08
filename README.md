# Multimodal Clinical Diagnosis System using Custom Deep Learning Architectures

This project starts with a chest X-ray classifier built from scratch using a custom CNN.

## Chest X-ray quick start

- Recommended test dataset: NIH ChestXray14 extracted to `~/Downloads/archive`
- The trainer automatically reads `Data_Entry_2017.csv` and the nested `images_###/images/` folders in that archive.
- For fast testing, the internal subset is capped at 200 images by default.
- That subset is split inside the script into 70% train, 15% validation, and 15% test.

## Run the first experiment

1. Install dependencies:
	- `pip install -r requirements.txt`
2. Train the model:
	- `python train_chest_xray.py --data-root ~/Downloads/archive --max-images 200 --epochs 1`

## Useful options

- `--data-root`: point this to a different chest X-ray dataset folder if needed.
- `--max-images`: keep this at 200 for quick iteration, or raise it later for fuller training.
- `--epochs`: increase after the first sanity check run.

## Output

- Model checkpoint: `outputs/chest_xray_cnn.pt`
- Class metadata: `outputs/chest_xray_classes.json`
- Test metrics summary: `outputs/test_metrics.json`
- Annotated preview images: `outputs/test_previews/`

The terminal output now includes:

- prediction confidence scores
- medical interpretation labels: `True Positive`, `False Positive`, `False Negative`, `True Negative`
- confusion matrix and test metrics: precision, recall, F1-score, and accuracy
- Grad-CAM overlays for the first five test images
