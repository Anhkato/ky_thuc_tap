using HeThongQuanLyHocVien.Models;
using HeThongQuanLyHocVien.Services;
using System.Collections.Generic;
using System.Linq;

namespace HeThongQuanLyHocVien.Services
{
    public class DichVuHocVien
    {
        private List<HocVien> _danhSachHocVien;
        private List<KhoaHoc> _danhSachKhoaHoc;

        public DichVuHocVien()
        {
            _danhSachKhoaHoc = new List<KhoaHoc>
            {
                new KhoaHoc("Lập trình C# Cơ bản", 40, CapDoKhoaHoc.CoBan),
                new KhoaHoc("Phát triển .NET Nâng cao", 60, CapDoKhoaHoc.NangCao),
                new KhoaHoc("Phát triển Web với ASP.NET Core", 80, CapDoKhoaHoc.TrungCap),
                new KhoaHoc("Cơ sở dữ liệu (SQL)", 30, CapDoKhoaHoc.CoBan)
            };
            _danhSachHocVien = XuLyTapTin.TaiHocVienTuTapTin(_danhSachKhoaHoc);
        }

        public void ThemHocVien(string ho, string ten, string email)
        {
            try
            {
                HocVien hocVienMoi = new HocVien(ho, ten, email);
                _danhSachHocVien.Add(hocVienMoi);
                Console.WriteLine($"Đã thêm học viên: {hocVienMoi.HoTen} (Mã: {hocVienMoi.MaHocVien})");
            }
            catch (ArgumentException ex)
            {
                Console.WriteLine($"Lỗi khi thêm học viên: {ex.Message}");
            }
        }

        public void DangKyKhoaHocChoHocVien(int maHocVien, int maKhoaHoc)
        {
            var hocVien = _danhSachHocVien.FirstOrDefault(hv => hv.MaHocVien == maHocVien);
            var khoaHoc = _danhSachKhoaHoc.FirstOrDefault(kh => kh.MaKhoaHoc == maKhoaHoc);

            if (hocVien == null)
            {
                Console.WriteLine($"Lỗi: Không tìm thấy học viên với mã {maHocVien}.");
                return;
            }
            if (khoaHoc == null)
            {
                Console.WriteLine($"Lỗi: Không tìm thấy khóa học với mã {maKhoaHoc}.");
                return;
            }

            hocVien.DangKyKhoaHoc(khoaHoc);
        }

        public void NhapDiemHocVien(int maHocVien, int maKhoaHoc, double diemSo)
        {
            var hocVien = _danhSachHocVien.FirstOrDefault(hv => hv.MaHocVien == maHocVien);
            var khoaHoc = _danhSachKhoaHoc.FirstOrDefault(kh => kh.MaKhoaHoc == maKhoaHoc);

            if (hocVien == null)
            {
                Console.WriteLine($"Lỗi: Không tìm thấy học viên với mã {maHocVien}.");
                return;
            }
            if (khoaHoc == null)
            {
                Console.WriteLine($"Lỗi: Không tìm thấy khóa học với mã {maKhoaHoc}.");
                return;
            }

            try
            {
                hocVien.LamBaiThi(khoaHoc, diemSo);
            }
            catch (ArgumentOutOfRangeException ex)
            {
                Console.WriteLine($"Lỗi nhập điểm: {ex.Message}");
            }
        }

        public List<KhoaHoc>? LayKhoaHocDaDangKyCuaHocVien(int maHocVien)
        {
            var hocVien = _danhSachHocVien.FirstOrDefault(hv => hv.MaHocVien == maHocVien);
            if (hocVien == null)
            {
                return null; 
            }
            // Trả về danh sách các khóa học đã đăng ký của học viên đó
            return hocVien.CacKhoaHocDaDangKy.Select(d => d.KhoaHoc).ToList();
        }


        public void HienThiTatCaHocVien()
        {
            if (_danhSachHocVien.Count == 0)
            {
                Console.WriteLine("Chưa có học viên nào trong hệ thống.");
                return;
            }
            Console.WriteLine("\n--- DANH SÁCH TẤT CẢ HỌC VIÊN ---");
            foreach (var hocVien in _danhSachHocVien)
            {
                hocVien.HienThiThongTin();
            }
            Console.WriteLine("----------------------------------");
        }

        public void HienThiTatCaKhoaHoc()
        {
            if (_danhSachKhoaHoc.Count == 0)
            {
                Console.WriteLine("Chưa có khóa học nào.");
                return;
            }
            Console.WriteLine("\n--- DANH SÁCH CÁC KHÓA HỌC HIỆN CÓ ---");
            foreach (var khoaHoc in _danhSachKhoaHoc)
            {
                khoaHoc.HienThiThongTinKhoaHoc();
            }
            Console.WriteLine("------------------------------------");
        }

        public void LuuDuLieu()
        {
            XuLyTapTin.LuuHocVienVaoTapTin(_danhSachHocVien);
        }
    }
}
