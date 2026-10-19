# Web test grayscale fine-tune

Trang GitHub Pages dùng checkpoint fine-tune tốt nhất epoch 8, khởi tạo từ checkpoint grayscale v3 epoch 32.

## Chạy thử local

Phục vụ thư mục này bằng static server, rồi mở `http://localhost:8000/`:

```bash
python -m http.server 8000
```

## Đưa lên GitHub Pages

Đưa toàn bộ nội dung thư mục này lên thư mục publish của GitHub Pages. Mở `index.html`; trình duyệt chạy ONNX Runtime Web + WASM. Ảnh không được gửi lên server.

## Tính năng

- Chọn ảnh đơn hoặc cả thư mục.
- Mở camera, căn giấy trong khung A4 nét đứt, chụp và chạy nhận dạng. Ảnh chụp được cắt theo khung căn trước khi đưa vào pipeline.
- Luôn chuyển ảnh sang grayscale trước khi tìm ROI và phân loại.
- Hiện ảnh grayscale, mask, ROI, confidence, logits, embedding, cosine từng lớp, thời gian, confusion matrix và chỉ số theo lớp.
- Lọc ảnh nhận dạng sai hoặc bị từ chối; xuất CSV.

## Model

- `models/roi_segmenter.onnx`: mask logits `[1,1,256,256]`.
- `models/roi_classifier.onnx`: logits 9 lớp và embedding 576 chiều.
- `models/prototypes.json`: prototype embedding và ngưỡng gợi ý.
- `models/metadata.json`: tiền xử lý, class IDs, kích thước tensor và chỉ số test.

Thứ tự ID: `0 bangtotnghiep`, `1 cccd`, `2 dtut`, `3 gxnkqthi`, `4 hocba10`, `5 hocba11`, `6 hocba12`, `7 hocbabia`, `8 vsat`.

Camera yêu cầu quyền truy cập và trang chạy trong secure context (HTTPS hoặc localhost). Ảnh camera và ảnh tải lên đều được chuyển grayscale trong trình duyệt trước khi inference.

Confidence và cosine prototype mặc định đều là `0.39`; có thể điều chỉnh trên giao diện. Ngưỡng thấp làm hệ thống dễ chấp nhận ảnh hơn và chưa được hiệu chuẩn bằng ảnh ngoài 9 lớp.

Test gồm 198 ảnh: pipeline accuracy 92.42%, paired ROI accuracy 99.49%, mask IoU 63.90%. Ngưỡng từ chối dựa trên tập validation chỉ có ảnh giấy tờ; cần hiệu chuẩn với ảnh ngoài 9 lớp trước khi dùng để khẳng định ảnh không phải giấy tờ.
