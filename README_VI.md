# Web tĩnh kiểm tra mô hình ROI giấy tờ

Gói này chạy toàn bộ nhận diện trong trình duyệt bằng ONNX Runtime Web. Không cần Python, API hoặc máy chủ inference. Ảnh người dùng chọn không được gửi khỏi trình duyệt.

## Đưa lên GitHub Pages

1. Tạo một repository GitHub mới.
2. Chép **toàn bộ nội dung** thư mục `handoff_roi_github` vào thư mục gốc repository, giữ nguyên các thư mục `models/` và `vendor/`.
3. Commit và push lên nhánh `main`.
4. Vào **Settings → Pages**, chọn deploy từ nhánh `main`, thư mục `/(root)`, rồi lưu.
5. Mở đường dẫn GitHub Pages được GitHub cấp. Dùng Chrome hoặc Edge bản mới.

Có thể triển khai bằng Git:

```bash
git init
git add .
git commit -m "Add static ONNX document demo"
git branch -M main
git remote add origin https://github.com/USERNAME/REPOSITORY.git
git push -u origin main
```

Không mở `index.html` trực tiếp bằng `file://`; trình duyệt chặn tải ONNX/WebAssembly theo cách đó. Dùng GitHub Pages hoặc máy chủ tĩnh. Để chạy thử tại máy:

```bash
python -m http.server 8000
```

Sau đó mở `http://localhost:8000` ngay trong thư mục gói.

## Cách chạy

- **Chọn ảnh** để dự đoán một ảnh.
- **Chọn thư mục** để duyệt nhiều ảnh. Ở Chrome/Edge, chọn thư mục có ảnh; nếu thư mục cha hoặc thư mục con trùng tên nhãn, trang lấy tên đó làm nhãn thật và tính accuracy/confusion matrix.
- Bấm **Chạy nhận diện**. Kết quả gồm ảnh, nhãn dự đoán, confidence, ROI fallback nếu mask không tạo được vùng, và thời gian xử lý mỗi ảnh.
- **Xuất CSV** tải bảng dự đoán và thời gian chạy xuống máy.

Ma trận chỉ tính các ảnh có tên một thư mục khớp chính xác một trong 9 tên lớp dưới đây. Chọn thư mục `test` chứa các thư mục lớp là cách thuận tiện nhất.

## Pipeline ONNX

Trang dùng hai nhánh ONNX của cùng checkpoint, theo thứ tự:

1. Ảnh gốc được xoay theo EXIF, giữ nguyên tỷ lệ và letterbox nền trắng về RGB `256×256`.
2. Chuẩn hóa RGB theo ImageNet: mean `(0.485, 0.456, 0.406)`, std `(0.229, 0.224, 0.225)`.
3. `roi_segmenter.onnx` tạo mask. Sigmoid và ngưỡng `0.45` tạo hình chữ nhật ROI; tọa độ được ánh xạ lại ảnh gốc, thêm biên 4% ngang và 8% dọc.
4. ROI ảnh gốc được letterbox về RGB `224×224`; `roi_classifier.onnx` cho 9 logits và trang áp dụng softmax.
5. CCCD được huấn luyện bằng ảnh xám. Vì chưa biết nhãn trước khi phân loại, trang chạy RGB trước; nếu nhánh đầu đoán CCCD, trang chạy lại segmenter trên ảnh xám rồi phân loại ROI xám. Trường hợp này có thêm một lượt inference và có thể chậm hơn.

Hai file ONNX là hai phần cần thiết của một pipeline tìm ROI rồi phân loại. Tách chúng cho phép web chỉ chạy segmenter một lần ở đường đi thông thường.

## Class ID

| ID | Tên lớp |
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

## Model và giới hạn kết quả

- Kiến trúc: MobileNetV3-Small encoder + U-Net ROI + classifier 9 lớp.
- Dùng best checkpoint epoch 27 từ `roi_all_classes_small_gray`; đợt train được dừng trong epoch 28 theo yêu cầu.
- Số liệu validation của checkpoint: pipeline accuracy **83.70%**, accuracy trên crop đã gán sẵn **93.33%**, mask IoU **71.26%**.
- Chưa có kết quả test split cho checkpoint này; các số validation không đại diện cho test thực tế.
- Inference dùng WebAssembly CPU, một luồng để tương thích với GitHub Pages. Thời gian hiển thị phụ thuộc máy/trình duyệt và bao gồm đọc ảnh, tạo tensor, chạy ONNX; lần đầu có thể chậm do khởi tạo WebAssembly.

## Nội dung gói

- `index.html`, `app.js`, `style.css`: giao diện GitHub Pages.
- `models/roi_segmenter.onnx`, `models/roi_classifier.onnx`: hai nhánh inference.
- `models/metadata.json`: kiến trúc, class ID, shapes, chuẩn hóa và pipeline.
- `vendor/`: ONNX Runtime Web 1.22.0 và WebAssembly đã đóng kèm, không cần CDN.
- `reports/`: biểu đồ loss/accuracy và confusion matrix validation tương ứng checkpoint.

Gói này không chứa `.pth` hoặc dữ liệu huấn luyện vì trang tĩnh chỉ cần ONNX.
