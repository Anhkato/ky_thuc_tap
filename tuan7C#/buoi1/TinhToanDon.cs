namespace UngDungBanBanh
{
    public static class TinhToanDon
    {
        public static double TinhTien(string tenBanh, int soLuong)
        {
            if (DuLieuBanh.GiaBanh.TryGetValue(tenBanh, out double donGia))
            {
                return donGia * soLuong;
            }
            return 0;
        }

        public static double TinhTien(double gia, int soLuong)
        {
            return gia * soLuong;
        }

        public static string PhanLoaiDon(double tongTien)
        {
            bool laDonLon = tongTien > 100000;
            return laDonLon ? "Đơn lớn" : "Đơn thường";
        }
    }
}