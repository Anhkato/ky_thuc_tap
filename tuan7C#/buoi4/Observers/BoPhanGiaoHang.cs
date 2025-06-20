using System;

namespace FoodDeliverySystem_VN
{
    public class BoPhanGiaoHang
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
            if (e.TrangThaiMoi == "Sẵn sàng giao")
            {
                Console.WriteLine($"[GIAO HÀNG] {e.DonHang} đã sẵn sàng. Điều phối shipper đến lấy hàng.");
            }
            else if (e.TrangThaiMoi == "Đang giao")
            {
                Console.WriteLine($"[GIAO HÀNG] Shipper đã nhận {e.DonHang} và đang trên đường giao.");
            }
             else if (e.TrangThaiMoi == "Hoàn tất")
            {
                Console.WriteLine($"[GIAO HÀNG] {e.DonHang} đã được giao thành công.");
            }
        }
    }
}