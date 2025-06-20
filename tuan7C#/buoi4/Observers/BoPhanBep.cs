using System;

namespace FoodDeliverySystem_VN
{
    public class BoPhanBep
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
            if (e.TrangThaiMoi == "Mới tạo")
            {
                Console.WriteLine($"[BẾP] Nhận yêu cầu cho {e.DonHang}. Bắt đầu chuẩn bị món ăn.");
            }
            else if (e.TrangThaiMoi == "Hủy" && e.TrangThaiCu == "Mới tạo")
            {
                Console.WriteLine($"[BẾP] Ngừng chuẩn bị cho {e.DonHang} vì đã bị hủy.");
            }
        }
    }
}