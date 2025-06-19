namespace HeThongQuanLyHocVien.Models
{
    public class DangKyKhoaHoc
    {
        public HocVien HocVien { get; private set; }
        public KhoaHoc KhoaHoc { get; private set; }
        public double DiemSo { get; set; }

        public DangKyKhoaHoc(HocVien hocVien, KhoaHoc khoaHoc)
        {
            HocVien = hocVien;
            KhoaHoc = khoaHoc;
            DiemSo = 0;
        }
    }
}
