using System;

namespace FoodDeliverySystem_VN
{
    public class DichVuKhachHang
    {
        public void DangKy(DonHang donHang)
        {
            donHang.TrangThaiDonHangThayDoi += XuLySuKien;
        }

        public void HuyDangKy(DonHang donHang)
        {
            donHang.TrangThaiDonHangThayDoi -= XuLySuKien;
        }

        private void XuLySuKien(object sender, ThongTinSuKienDonHang e)
        {
            if (e.TrangThaiMoi == "Hủy")
            {
                Console.WriteLine($"[CSKH] Ghi nhận {e.DonHang} đã bị hủy. Hệ thống sẽ xử lý hoàn tiền nếu cần.");
            }
            else if (e.TrangThaiMoi == "Giao thất bại")
            {
                Console.WriteLine($"[CSKH] {e.DonHang} giao không thành công. Nhân viên sẽ liên hệ lại với khách hàng.");
            }
        }
    }
}