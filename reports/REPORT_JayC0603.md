# BÁO CÁO CẢI TIẾN ỨNG DỤNG POSEFORGE

**Tác giả:** JayC0603  
**Ngày báo cáo:** 02/11/2026  
**Dự án:** PoseForge – Nhận diện tư thế trong lớp học  
**Phạm vi:** Giao diện desktop, cấu hình CPU và bố cục hiển thị camera.

## 1. Mục tiêu

Cải thiện giao diện ứng dụng để việc chọn nguồn hình ảnh, chạy nhận diện và theo dõi kết quả thuận tiện hơn. Ứng dụng cần sử dụng được trên máy chạy CPU, đồng thời giữ các chức năng hiện có như nhiều nguồn hình ảnh, lựa chọn mô hình, ROI và xuất kết quả.

## 2. Vấn đề ban đầu

- Các nút cấu hình tập trung phía trên khiến bố cục khó phân biệt giữa thiết lập và theo dõi nhận diện.
- Cấu hình YOLO mặc định sử dụng `cuda:0`. Trên máy không có CUDA, quá trình khởi tạo phát sinh lỗi `Invalid CUDA device` và không chạy được nhận diện.
- Khối xác suất và các thành phần giới thiệu chiếm nhiều chiều cao, khiến khung camera bị tràn khỏi vùng xem. Người dùng phải cuộn để quan sát đầy đủ.
- Cần làm rõ cách nhập nguồn camera từ DroidCam.

## 3. Các thay đổi đã thực hiện

### 3.1. Thiết kế lại giao diện

Giao diện chuyển sang bố cục thanh bên trái và vùng quan sát bên phải. Thanh bên chứa nguồn hình ảnh, nút mở video, thêm/bớt nguồn, lựa chọn mô hình, chọn CPU và cài đặt xuất dữ liệu. Vùng chính hiển thị trạng thái phiên, nút bắt đầu/kết thúc và hình ảnh nhận diện.

Thiết kế sử dụng nền tối, điểm nhấn tím, các khối bo góc và phân cấp chữ rõ hơn. Các nút chính cùng một phần nội dung cấu hình đã được Việt hóa; một số thông báo kỹ thuật vẫn giữ tiếng Anh.

### 3.2. Hỗ trợ CPU làm mặc định

- Đổi thiết bị YOLO mặc định từ `cuda:0` sang `cpu`.
- Đồng bộ trạng thái nút CPU với cấu hình ban đầu.
- Nút CPU luôn chọn CPU khi nhấn; nhấn lần nữa không chuyển ngược sang CUDA.
- Có thể chủ động chọn CUDA trong cài đặt nâng cao nếu máy có môi trường phù hợp.
- Bộ phân loại tiếp tục chạy trên CPU; các tùy chọn ONNX/PT hiện có được giữ lại.

Thay đổi này xử lý nguyên nhân cấu hình gây lỗi CUDA trên máy dùng CPU. Hiệu năng nhận diện thực tế trên CPU chưa được đo trong đợt thay đổi này.

### 3.3. Hướng dẫn nguồn DroidCam

Thêm hướng dẫn tại ô nhập nguồn: sử dụng chỉ số camera ảo DroidCam trên Windows hoặc URL HTTP/RTSP do ứng dụng cung cấp. Đây là cách sử dụng các loại nguồn đã được hỗ trợ; không bổ sung cơ chế tự dò thiết bị hay tự tạo URL điện thoại.

Kết nối DroidCam thực tế chưa được xác minh. Người dùng cần kiểm tra camera ảo hoặc cung cấp URL luồng chính xác, đồng thời bảo đảm máy tính truy cập được nguồn hình ảnh.

### 3.4. Sửa khung camera bị tràn

- Ẩn các thẻ giới thiệu khi bắt đầu phiên nhận diện để dành thêm diện tích cho camera.
- Đưa bảng xác suất vào vùng cuộn riêng, tránh làm tăng chiều cao bắt buộc của toàn bộ khung camera.
- Điều chỉnh chính sách kích thước của vùng cuộn xác suất để khung hình co giãn theo diện tích hiện có.
- Mặc định thu gọn chi tiết hiệu suất; người dùng vẫn có thể bật lại.
- Đưa thanh cuộn vùng quan sát về đầu khi bắt đầu phiên mới.

Kiểm thử một nguồn ở cửa sổ 1100 × 700 xác nhận khung xem nằm trọn trong vùng quan sát và không cần cuộn dọc. Với nhiều nguồn hoặc cửa sổ nhỏ hơn, vùng quan sát vẫn có thể cần cuộn.

## 4. Chức năng được giữ lại

| Nhóm | Chức năng |
| --- | --- |
| Nguồn hình ảnh | Webcam theo chỉ số, video cục bộ, luồng HTTP/RTSP; tối đa 4 nguồn |
| Mô hình | Chọn các mô hình có bundle hợp lệ; chọn backend ONNX hoặc PT theo khả năng của bundle |
| Phiên nhận diện | Bắt đầu, kết thúc, khóa cấu hình khi chạy và giải phóng nguồn |
| Kết quả | Nhãn tư thế, độ tin cậy, xác suất, thông tin pose và tracking |
| ROI | Chọn vùng, bật/tắt và đặt lại riêng cho từng nguồn |
| Hiệu suất | FPS YOLO, bộ phân loại, pipeline, hiển thị và chi tiết thời gian |
| Xuất dữ liệu | Tùy chọn video MP4 có chú thích, CSV/JSONL và báo cáo phiên |

## 5. Tệp thay đổi

| Tệp | Vai trò |
| --- | --- |
| `inference/gui.py` | Tích hợp giao diện, cấu hình CPU và sửa bố cục khung camera |
| `inference/workspace.py` | Tổ chức thanh bên, vùng quan sát và màn hình chờ |
| `inference/theme.py` | Định nghĩa màu sắc, kiểu chữ và kiểu hiển thị widget |
| `tests/test_inference_desktop.py` | Kiểm thử giao diện, nút CPU và khung xem ở kích thước laptop |

Phạm vi chỉnh sửa không bao gồm huấn luyện lại mô hình, thay checkpoint hoặc thay đổi thuật toán nhận diện.

## 6. Kiểm thử và kết quả

Lệnh kiểm thử đã chạy từ thư mục gốc của mã nguồn trên Windows:

```powershell
$env:QT_QPA_PLATFORM = 'windows'
.\.venv\Scripts\python.exe -m pytest tests/test_inference_desktop.py -q
```

**Kết quả gần nhất: 19 passed in 10.87s.**

Các nội dung được kiểm tra gồm khởi tạo không tự mở camera, cấu hình CPU mặc định, nút CPU không tự chuyển sang CUDA, bắt đầu/dừng và khởi động lại phiên, nhiều nguồn độc lập, ánh xạ ROI, vòng đời tiến trình và bố cục một nguồn ở cửa sổ 1100 × 700. Giao diện Windows cũng đã được mở để kiểm tra trực quan trong quá trình thiết kế.

Các kiểm thử giao diện sử dụng nguồn giả lập khi phù hợp. Kết quả trên không xác nhận chất lượng nhận diện, FPS camera thực tế hay khả năng kết nối DroidCam trên thiết bị của người dùng. Toàn bộ bộ kiểm thử của dự án chưa được chạy lại trong đợt thay đổi này.

## 7. Cách chạy

Mở PowerShell và thực hiện:

```powershell
cd C:\Users\nftvi\Downloads\PoseForge_Reposory\PoseForge_Reposory
.\.venv\Scripts\python.exe -m inference.gui
```

Đóng cửa sổ ứng dụng cũ trước khi kiểm tra bản mới. Thiết bị mặc định là CPU. Chọn nguồn webcam, video hoặc địa chỉ luồng, sau đó nhấn **Bắt đầu**. Với DroidCam, nhập chỉ số camera ảo đã xác định hoặc URL luồng chính xác. Nhấn **Kết thúc** trước khi đổi mô hình, thiết bị hoặc nguồn hình ảnh.

## 8. Công việc cần xác minh trên thiết bị

1. Kết nối DroidCam và kiểm tra nguồn hình ảnh được mở thành công.
2. Kiểm tra khung hình, hướng ảnh, keypoint và ROI khi người dùng di chuyển.
3. Theo dõi FPS trên CPU với một nguồn trước khi tăng số nguồn.
4. Bật các tùy chọn xuất cần dùng, kết thúc phiên và kiểm tra tệp trong `artifacts/inference_runs/`.
5. Kiểm tra bố cục với độ phân giải và mức phóng đại màn hình thực tế của người dùng.
