using System;
using System.Collections.Generic;

namespace FoodDeliverySystem_VN
{
    public class DonHang
    {
        public int MaDonHang { get; }
        public List<string> CacMon { get; }
        private string _trangThai;

        public event EventHandler<ThongTinSuKienDonHang> TrangThaiDonHangThayDoi;

        public DonHang(int maDonHang, List<string> cacMon)
        {
            MaDonHang = maDonHang;
            CacMon = cacMon;
            _trangThai = "Chưa khởi tạo";
        }

        public string TrangThai
        {
            get { return _trangThai; }
        }

        public void CapNhatTrangThai(string trangThaiMoi)
        {
            string trangThaiCu = _trangThai;
            _trangThai = trangThaiMoi;
            KichHoatSuKienThayDoiTrangThai(new ThongTinSuKienDonHang(this, trangThaiCu, trangThaiMoi));
        }

        protected virtual void KichHoatSuKienThayDoiTrangThai(ThongTinSuKienDonHang e)
        {
            TrangThaiDonHangThayDoi?.Invoke(this, e);
        }

        public override string ToString()
        {
            return $"Đơn hàng #{MaDonHang} [{string.Join(", ", CacMon)}]";
        }
    }
}