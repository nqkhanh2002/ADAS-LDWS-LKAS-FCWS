<a name="readme-top"></a>
<div align="center">
  <p align="center">
    <a href="https://www.youtube.com/channel/UCKaMI0RBxF26f6j0Q8RRyTw">View Demo</a>
    ·
    <a href="https://github.com/nqkhanh2002/Lane-Detection-for-Self-Driving-Cars/issues">Report Bug</a>
    ·
    <a href="https://github.com/nqkhanh2002/Lane-Detection-for-Self-Driving-Cars/pulls">Request Feature</a>
  </p>
</div>

<h1 align="center">Lane-Detection-for-Self-Driving-Cars</h1>
<img src="Image_Resrouces\intro_readme.png"> <br>

Phát triển một giải pháp phát hiện và theo dõi làn đường cho xe tự lái. Đây là một phần quan trọng của hệ thống lái tự động, nhằm đảm bảo an toàn và hiệu quả cho người lái và hành khách trên đường.
# Thành viên nhóm 
| STT | Họ và tên | MSSV |
|-------|-------|-------|
| 1 | Nguyễn Quốc Khánh | 20521452 |
| 2 | Đinh Phương Nam | 20520641 |
# Cấu trúc dự án
# Phương pháp
Bài toán triển khai theo 2 cách:
1. Phương pháp xử lý ảnh truyền thống
2. Phương pháp mạng nơ-ron học sâu
3. Xây dựng giao diện người dùng (GUI) để sử dụng và so sánh 2 phương pháp 
------- 
## Phương pháp xử lý ảnh truyền thống
Pipline:
1. Tính toán ma trận hiệu chỉnh camera (camera calibration matrix) và hệ số méo hình ảnh (distortion coefficients).

Trong quá trình chụp ảnh, các thông số của máy ảnh có thể gây ra sự biến dạng trong hình ảnh. Điều này có thể ảnh hưởng đến khả năng phát hiện và đo đạc các đối tượng trên hình ảnh, ví dụ như đường viền của một chiếc xe.

Do đó, để có thể giải quyết vấn đề này, ta cần phải tính toán ma trận hiệu chỉnh camera và hệ số méo hình ảnh để sửa chữa các biến dạng trong hình ảnh gốc. Trong dự án này, quá trình này được thực hiện bằng cách sử dụng các hình ảnh mẫu của một bàn cờ cờ vua ở các góc độ khác nhau (Được đặt trong folder Camera_Calibration_Chessboard) để ước tính ma trận hiệu chỉnh và hệ số méo hình ảnh.

Khi đã tính toán được ma trận hiệu chỉnh camera và hệ số méo hình ảnh, ta sẽ áp dụng chúng vào hình ảnh gốc để sửa chữa các biến dạng của hình ảnh. Quá trình này giúp cho các đối tượng trên hình ảnh được tái hiện một cách chính xác hơn và cải thiện khả năng phát hiện và đo đạc chính xác các đối tượng trên hình ảnh.

2. Áp dụng sự hiệu chỉnh méo cho ảnh gốc.
3. Sử dụng các chuyển đổi màu sắc, độ dốc, vv, để tạo ra một hình ảnh nhị phân được ngưỡng.
4. Áp dụng phép chuyển đổi góc nhìn để tạo ra một "góc nhìn chim" của hình ảnh.
5. Phát hiện các pixel của làn đường và phù hợp để tìm ranh giới của làn đường.
6. Xác định độ cong của làn đường và vị trí xe so với trung tâm.
7. Chuyển đổi lại ranh giới của làn đường được phát hiện trở lại hình ảnh ban đầu và hiển thị ước tính số liệu của độ cong của làn đường và vị trí xe.

