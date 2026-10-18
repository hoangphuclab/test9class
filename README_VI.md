# ROI document classifier

## Deploy on GitHub Pages

1. Put the contents of this folder at the repository root.
2. Push the repository to GitHub.
3. In **Settings → Pages**, deploy from branch `main`, folder `/(root)`.
4. Open the published URL in Chrome or Edge. To test locally, run `python -m http.server 8000` from this folder and open `http://localhost:8000`.

## Use

- Select one image or a folder, then run prediction.
- Folder names matching a class are used as labels for accuracy and the confusion matrix.
- Each result shows the predicted class, confidence, ROI, timing, all class probabilities, 576D embedding, and cosine similarity to each class prototype.
- The JSON output download includes raw mask logits, classifier logits, embedding, and prototype cosine scores.

## Model outputs

This is one trained MobileNetV3-Small + U-Net model exported as two ONNX graph files because the browser must crop the ROI between segmentation and classification:

- `models/roi_segmenter.onnx`: input `page [1,3,256,256]`; output `mask_logits [1,1,256,256]`.
- `models/roi_classifier.onnx`: input `roi [1,3,224,224]`; outputs `class_logits [1,9]` and `embedding [1,576]`.
- `models/prototypes.json`: normalized mean embedding for each training class. Compare output embeddings with these using cosine similarity.

The page runs the RGB pipeline first. If its first prediction is `cccd`, it reruns segmentation and classification after converting the page and ROI to grayscale and replicating gray into three channels. Other labels remain RGB. CCCD training, validation, and test samples also use grayscale; there is no thresholding/binarization.

Class IDs:

| ID | Label |
|---:|---|
| 0 | `bangtotnghiep` |
| 1 | `cccd` |
| 2 | `dtut` |
| 3 | `gxn` |
| 4 | `hocba10` |
| 5 | `hocba11` |
| 6 | `hocba12` |
| 7 | `hocbabia` |
| 8 | `vsat` |

## Training snapshot

- Dataset: 872 cleaned original/crop pairs; split 70/15/15 by class, grouped by source-image hash.
- Best checkpoint: epoch 15, MobileNetV3-Small + U-Net, 2,285,682 parameters.
- Test set: 130 images; pipeline accuracy 90.77%, paired-crop accuracy 98.46%, mask IoU 76.48%.
- Full training curves, per-class metrics, confusion matrices, predictions, and embedding/cosine analysis are in `reports/`.

Cosine values rank similarity to the nine known document prototypes. The dataset has no dog/cat or other non-document negatives, so these values are not a calibrated unknown-document rejection test. Include representative non-document samples and calibrate a threshold before relying on automatic rejection.

The web app runs ONNX Runtime Web and WebAssembly locally in the browser. It does not send selected images to a server and does not require Python for deployment.
