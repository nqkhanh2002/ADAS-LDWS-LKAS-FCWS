1. Paper triển khai: https://www.researchgate.net/publication/332298502_A_Computer_Vision_based_Lane_Detection_Approach
2. Code chính của paper: https://github.com/Dt-Pham/Advanced-Lane-Lines
3. https://github.com/uppala75/CarND-Advanced-Lane-Lines/blob/master/AdvancedLaneLines.md
4. https://towardsdatascience.com/advanced-lane-detection-for-autonomous-vehicles-using-computer-vision-techniques-f229e4245e41


Các kết quả thử không thành công đối với video 2 làn trắng
1. https://github.com/georgesung/advanced_lane_detection
Không gian màu HLS tốt hơn không gian màu BGR để phát hiện các vấn đề về hình ảnh do ánh sáng, chẳng hạn như bóng, ánh sáng chói từ mặt trời, đèn pha, v.v. Chúng tôi muốn loại bỏ tất cả những điều này để giúp phát hiện các vạch làn đường dễ dàng hơn. Vì lý do này, chúng tôi sử dụng không gian màu HLS, phân chia tất cả các màu thành các giá trị sắc độ, độ bão hòa và độ sáng.