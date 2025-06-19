namespace HeThongQuanLyHocVien.Models
{
    public class KhoaHoc
    {
        private static int _maKhoaHocTiepTheo = 200;
        public int MaKhoaHoc { get; private set; }
        public string TenKhoaHoc { get; private set; }
        public int ThoiLuongGio { get; private set; }
        public CapDoKhoaHoc CapDo { get; private set; }

        public KhoaHoc(string tenKhoaHoc, int thoiLuongGio, CapDoKhoaHoc capDo)
        {
            MaKhoaHoc = _maKhoaHocTiepTheo++;
            TenKhoaHoc = tenKhoaHoc;
            ThoiLuongGio = thoiLuongGio;
            CapDo = capDo;
        }

        public KhoaHoc(int maKhoaHoc, string tenKhoaHoc, int thoiLuongGio, CapDoKhoaHoc capDo)
        {
            MaKhoaHoc = maKhoaHoc;
            if (maKhoaHoc >= _maKhoaHocTiepTheo)
            {
                _maKhoaHocTiepTheo = maKhoaHoc + 1;
            }
            TenKhoaHoc = tenKhoaHoc;
            ThoiLuongGio = thoiLuongGio;
            CapDo = capDo;
        }

        public void HienThiThongTinKhoaHoc()
        {
            Console.WriteLine($"Mã khóa học: {MaKhoaHoc}, Tên: {TenKhoaHoc}, Thời lượng: {ThoiLuongGio} giờ, Cấp độ: {CapDo}");
        }
    }
}
