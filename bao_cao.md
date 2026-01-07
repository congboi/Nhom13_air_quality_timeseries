Q1. Kiểm tra mức độ hiểu dữ liệu & EDA chuỗi thời gian
1. Kiểm tra khoảng thời gian và tính liên tục của dữ liệu

Khoảng thời gian:
Dữ liệu PM2.5 được sử dụng trong nghiên cứu có tần suất theo giờ, bao phủ từ 2013-03-01 00:00:00 đến 2017-02-28 23:00:00.

Xác nhận tính liên tục:
Qua thống kê số lượng bản ghi theo trạm, mỗi trạm quan trắc đều có 35.064 dòng dữ liệu, tương ứng đầy đủ số giờ trong khoảng thời gian nghiên cứu.

Kết luận:
Với số lượng bản ghi khớp tuyệt đối giữa các trạm trên cùng một khoảng thời gian, có thể khẳng định dữ liệu chuỗi thời gian theo giờ là liên tục, không bị mất mốc thời gian, đảm bảo điều kiện cần cho các mô hình chuỗi thời gian và hồi quy dự báo.

2. Tỷ lệ dữ liệu thiếu và quan sát theo thời gian

Tỷ lệ dữ liệu thiếu:
Phân tích cho thấy một số biến có tỷ lệ thiếu đáng kể, trong đó:

CO là biến có tỷ lệ thiếu cao nhất (xấp xỉ 5%)

O3 có tỷ lệ thiếu khoảng 3%

PM2.5 – biến mục tiêu – có tỷ lệ thiếu ở mức hơn 2%

Quan sát:
Các giá trị thiếu không phân bố ngẫu nhiên mà có xu hướng tập trung tại một số cụm thời gian nhất định. Khi xây dựng các đặc trưng trễ (lag features) cho PM2.5, các giá trị thiếu này có thể gây ra hiệu ứng lan truyền, làm tăng số dòng dữ liệu không hợp lệ ở các thời điểm kế tiếp.

3. Ngoại lai (Outliers) và phân phối dữ liệu

Phân phối dữ liệu:
Phân phối của PM2.5 thể hiện sự lệch phải rất mạnh, cho thấy phần lớn các quan sát rơi vào mức ô nhiễm trung bình và cao, trong khi mức ô nhiễm thấp chiếm tỷ lệ nhỏ.

Ngoại lai:
Chuỗi thời gian xuất hiện nhiều đỉnh nhọn (spikes) với giá trị rất lớn, trong một số thời điểm vượt quá 600. Đây là các giá trị cực đoan có ý nghĩa thực tế nhưng đồng thời cũng là nguyên nhân chính làm gia tăng sai số dự báo của mô hình, đặc biệt là chỉ số RMSE.

4. Trực quan hóa chuỗi thời gian PM2.5

Toàn bộ giai đoạn:
Chuỗi PM2.5 thể hiện tính mùa vụ rõ rệt, với nồng độ ô nhiễm thường tăng cao vào các tháng mùa đông và giảm vào mùa hè.

Phóng to theo thời gian ngắn:
Khi quan sát trong khoảng thời gian ngắn (1–2 tháng), có thể nhận thấy chu kỳ ngày/đêm (24 giờ) rõ ràng, đồng thời xuất hiện các đợt ô nhiễm kéo dài từ 2 đến 3 ngày.

5. Kiểm tra tự tương quan và tính dừng của chuỗi

Tự tương quan:
Biểu đồ ACF cho thấy PM2.5 có mối tương quan mạnh với các giá trị trong quá khứ, đặc biệt tại độ trễ 24 giờ, phản ánh chu kỳ ngày đã được quan sát trong bước trực quan hóa.

Kiểm tra tính dừng:
Kết quả kiểm định ADF cho p-value nhỏ hơn 0.05, cho phép bác bỏ giả thuyết chuỗi không dừng. Do đó, chuỗi PM2.5 được xem là chuỗi dừng, phù hợp cho việc áp dụng mô hình ARIMA với tham số sai phân 
𝑑
=
0
d=0.

6. Biến đáng lo ngại nhất và nguyên nhân

Biến đáng lo ngại nhất: PM2.5

Lý do:
PM2.5 là biến mục tiêu cần dự báo. Việc thiếu dữ liệu tại biến này không chỉ ảnh hưởng trực tiếp đến quá trình huấn luyện mô hình mà còn gây ra hiệu ứng dây chuyền khi xây dựng các đặc trưng trễ, làm giảm đáng kể số lượng mẫu hợp lệ và độ chính xác dự báo.

Q2. Giải thích baseline hồi quy dự báo
1. Ý nghĩa của đặc trưng Lag 24h

Đặc trưng trễ 24 giờ đóng vai trò then chốt trong mô hình hồi quy dự báo PM2.5 do phản ánh chu kỳ sinh hoạt và khí tượng theo ngày. Các kết quả huấn luyện cho thấy mô hình với các đặc trưng trễ có khả năng giải thích phần lớn biến động của PM2.5 theo thời gian.

2. Tầm quan trọng của việc chia dữ liệu bằng mốc Cutoff

Dữ liệu được chia bằng mốc thời gian 2017-01-01 nhằm đảm bảo nguyên tắc nhân quả trong dự báo chuỗi thời gian. Cách chia này giúp tránh rò rỉ dữ liệu và phản ánh đúng kịch bản triển khai mô hình trong thực tế.

3. Phân biệt RMSE và MAE thông qua kết quả thực tế

Kết quả đánh giá cho thấy RMSE lớn hơn đáng kể so với MAE. Điều này xuất phát từ sự tồn tại của các đỉnh ô nhiễm lớn trong dữ liệu PM2.5, khiến các sai số lớn bị phạt nặng hơn trong công thức tính RMSE.

Q3. Quy trình ra quyết định ARIMA
1. Quan sát chuỗi gốc (Trend & Seasonality)

Chuỗi PM2.5 thể hiện nhiều biến động mạnh và có chu kỳ ngày rõ rệt. Tuy nhiên, mô hình ARIMA đơn biến chủ yếu mô phỏng được xu hướng trung bình của chuỗi.

2. Kiểm định tính dừng để lựa chọn tham số ( d )

Kết quả kiểm định ADF cho thấy chuỗi PM2.5 là chuỗi dừng, do đó lựa chọn tham số sai phân 
𝑑
=
0
d=0.

3. Quan sát ACF và PACF để gợi ý ( p, q )

ACF và PACF cung cấp cơ sở để lựa chọn các giá trị ứng viên cho 
𝑝
p và 
𝑞
q, phản ánh cấu trúc tự tương quan trong chuỗi.

4. Grid Search và lựa chọn mô hình tối ưu theo AIC

Mô hình ARIMA tối ưu được lựa chọn thông qua Grid Search với tiêu chí AIC, đảm bảo sự cân bằng giữa độ chính xác và độ phức tạp.

5. Kết quả dự báo và so sánh với mô hình hồi quy

Kết quả cho thấy mô hình ARIMA có sai số cao hơn so với mô hình hồi quy. Điều này cho thấy các biến ngoại sinh và đặc trưng trễ đóng vai trò quan trọng trong việc dự báo chính xác nồng độ PM2.5.