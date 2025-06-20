using System;

namespace FoodDeliverySystem_VN
{
    public class ThongTinSuKienDonHang : EventArgs
    {
        public DonHang DonHang { get; }
        public string TrangThaiCu { get; }
        public string TrangThaiMoi { get; }
        public DateTime ThoiDiem { get; }

        public ThongTinSuKienDonHang(DonHang donHang, string trangThaiCu, string trangThaiMoi)
        {
            DonHang = donHang;
            TrangThaiCu = trangThaiCu;
            TrangThaiMoi = trangThaiMoi;
            ThoiDiem = DateTime.Now;
        }
    }
}