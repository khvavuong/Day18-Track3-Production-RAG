# Chính sách An toàn Thông tin

## Chương 1 — Quản lý mật khẩu

### Quy tắc mật khẩu

Mật khẩu hệ thống nội bộ phải có tối thiểu 12 ký tự, bao gồm chữ hoa, chữ thường, số và ký tự đặc biệt. Người dùng phải thay đổi mật khẩu mỗi 90 ngày một lần. Hệ thống tự động khóa tài khoản sau 5 lần đăng nhập sai liên tiếp trong vòng 15 phút.

Không được dùng lại 5 mật khẩu gần nhất. Nghiêm cấm chia sẻ mật khẩu qua email, tin nhắn, hoặc bất kỳ kênh không mã hóa nào.

### Xác thực hai yếu tố

Tất cả tài khoản truy cập hệ thống quan trọng bắt buộc bật xác thực hai yếu tố (2FA) qua ứng dụng Microsoft Authenticator hoặc Google Authenticator. Không chấp nhận 2FA qua SMS do rủi ro SIM swap.

## Chương 2 — Truy cập từ xa (VPN)

### Cấu hình VPN

Hệ thống VPN sử dụng giao thức WireGuard với mã hóa AES-256-GCM và xác thực Curve25519. Mỗi nhân viên được cấp 1 cấu hình VPN riêng, gắn với MAC address của thiết bị làm việc.

### Quy định sử dụng

VPN chỉ được dùng trên thiết bị do công ty cấp hoặc đã được IT phê duyệt cài MDM. Phiên VPN tự động ngắt sau 8 giờ không hoạt động. Báo cáo IT ngay nếu nghi ngờ thiết bị bị mất hoặc xâm phạm.

## Chương 3 — Quản lý dữ liệu

### Phân loại dữ liệu

Dữ liệu được phân thành 4 cấp: Công khai, Nội bộ, Mật, Tuyệt mật. Dữ liệu cá nhân của khách hàng và nhân viên thuộc cấp Mật trở lên, phải tuân thủ Nghị định 13/2023/NĐ-CP về Bảo vệ Dữ liệu Cá nhân.

### Sao lưu

Dữ liệu hệ thống được sao lưu hằng ngày lúc 2:00 sáng vào hai trung tâm dữ liệu địa lý cách nhau tối thiểu 200 km. Lưu trữ tối thiểu 90 ngày cho dữ liệu nóng và 7 năm cho dữ liệu tài chính.

## Chương 4 — Phản ứng sự cố

### Báo cáo sự cố

Mọi sự cố an ninh (nghi ngờ phishing, malware, mất thiết bị, lộ dữ liệu) phải được báo cáo qua hotline IT 1900-xxxx hoặc email security@vinuni.edu.vn trong vòng 1 giờ kể từ khi phát hiện.

### Quy trình xử lý

Đội phản ứng sự cố (CSIRT) phản hồi trong 30 phút với sự cố mức cao. Mọi sự cố đều được điều tra và lập báo cáo hậu kỳ trong 7 ngày làm việc.
