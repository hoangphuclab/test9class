# ROI document classifier

## Deploy on GitHub Pages

1. Put the contents of this folder at the repository root.
2. Push the repository to GitHub.
3. In **Settings → Pages**, deploy from branch `main`, folder `/(root)`.
4. Open the published URL in Chrome or Edge. To test locally, run `python -m http.server 8000` from this folder and open `http://localhost:8000`.

## Use

- Select one image or a folder, then run prediction.
- Adjust **Confidence tối thiểu** and **Cosine prototype tối thiểu** on the page. The result changes immediately when either threshold moves.
- An image is accepted only if both scores meet their thresholds. Otherwise it is marked **TỪ CHỐI** and counted in the reject column of the confusion matrix.
- Starting thresholds are confidence `0.60` and cosine `0.64`, derived from low percentiles of validation document images. They are editable starting points, not calibrated dog/cat rejection limits.
- Folder names matching a class are used as labels for accuracy and the confusion matrix.
- The JSON output includes raw mask logits, classifier logits, the 576D embedding, per-class cosine scores, thresholds, and the accept/reject decision. CSV includes candidate, decision, confidence, and nearest prototype.

## Model outputs

This is one trained MobileNetV3-Small + U-Net model exported as two ONNX graph files because the browser must crop the ROI between segmentation and classification:

- `models/roi_segmenter.onnx`: input `page [1,3,256,256]`; output `mask_logits [1,1,256,256]`.
- `models/roi_classifier.onnx`: input `roi [1,3,224,224]`; outputs `class_logits [1,9]` and `embedding [1,576]`.
- `models/prototypes.json`: normalized mean embedding for each training class. Compare output embeddings with these using cosine similarity.

With a folder whose path contains the class directory, samples labeled `cccd` are converted to grayscale before the first segmentation and classification pass, matching the training preprocessing. For one unlabeled image, tick **Ảnh đơn là CCCD · grayscale** to force grayscale; otherwise the page runs RGB first and reruns in grayscale if its first prediction is `cccd`. There is no binarization.

## Threshold behavior and limitation

The UI rejects an image if either `max softmax confidence < confidence threshold` **or** `maximum cosine to the 9 document prototypes < cosine threshold`. You can tune both sliders against a folder with known labels and inspect the rejection column of the confusion matrix.

The current training data contains only the nine document classes, with no dog/cat or other non-document negatives. Therefore the default cutoffs can reject low-scoring inputs, but cannot guarantee that an arbitrary non-document will be rejected; some may still score highly on both measures. Add representative negative images and calibrate on a separate validation set before relying on this as an unknown detector.

## Training snapshot

- Dataset: 892 original/ROI pairs across 9 classes; split 70/15/15 by class, grouped by source-image hash (626 train, 133 validation, 133 test).
- Class order: `bangtotnghiep`, `cccd`, `dtut`, `gxnkqthi`, `hocba10`, `hocba11`, `hocba12`, `hocbabia`, `vsat`.
- Best checkpoint: epoch 17, MobileNetV3-Small + U-Net, 2,285,682 parameters.
- Test set: 133 images; pipeline accuracy 93.98%, paired-crop accuracy 100%, mask IoU 78.09%.
- CCCD images are converted to grayscale in all three splits. Folder testing uses the known class directory to apply grayscale to CCCD from the first pass; an unlabeled single image uses the predicted-CCCD fallback.
- Full training curves, per-class metrics, confusion matrices, predictions, and embedding/cosine analysis are in `reports/`.

The web app runs ONNX Runtime Web and WebAssembly locally in the browser. It does not send selected images to a server and does not require Python for deployment.

## Chọn vùng ROI

Mask đầu ra là một bản đồ xác suất `[1,1,256,256]`, không phải nhiều mask riêng. Web lấy các pixel có sigmoid từ `0.45`, gom thành phần liên thông 8 hướng, bỏ vùng dưới 12 pixel, rồi chỉ lấy vùng có confidence sigmoid trung bình cao nhất để tạo bbox và crop. Overlay và ảnh mask chỉ tô vùng được chọn; JSON vẫn giữ toàn bộ logits gốc.

Đánh giá lại checkpoint theo cách chọn này nằm trong `runs/roi_all_classes_gxnkqthi_gray_cccd/evaluation_top_component/`. Tập test đạt pipeline accuracy 93.98% (125/133 ảnh).
