namespace HeThongQuanLyHocVien.Models
{
    public interface IGiaoDienHocTap
    {
        void DangKyKhoaHoc(KhoaHoc khoaHoc);
        void LamBaiThi(KhoaHoc khoaHoc, double diemSo);
    }
}
