using System.Collections.Generic;
using System.Linq;

namespace HeThongQuanLyHocVien.Models
{
    public class HocVien : Nguoi, IGiaoDienHocTap
    {
        private static int _maHocVienTiepTheo = 1;
        public int MaHocVien { get; private set; }
        private double _diemTongKet;
        private string _trinhDo;

        public double DiemTongKet
        {
            get { return _diemTongKet; }
            set
            {
                if (value < 0 || value > 10)
                {
                    throw new ArgumentOutOfRangeException(nameof(DiemTongKet), "Điểm phải từ 0 đến 10.");
                }
                _diemTongKet = value;
                CapNhatTrinhDo();
            }
        }

        public string TrinhDo
        {
            get { return _trinhDo; }
            private set { _trinhDo = value; }
        }

        public List<DangKyKhoaHoc> CacKhoaHocDaDangKy { get; private set; }

        public HocVien(string ho, string ten, string email)
            : base(ho, ten, email)
        {
            MaHocVien = _maHocVienTiepTheo++;
            CacKhoaHocDaDangKy = new List<DangKyKhoaHoc>();
            _diemTongKet = 0;
            CapNhatTrinhDo();
        }

        public HocVien(int maHocVien, string ho, string ten, string email)
            : base(ho, ten, email)
        {
            MaHocVien = maHocVien;
            if (maHocVien >= _maHocVienTiepTheo)
            {
                _maHocVienTiepTheo = maHocVien + 1;
            }
            CacKhoaHocDaDangKy = new List<DangKyKhoaHoc>();
            _diemTongKet = 0;
            CapNhatTrinhDo();
        }

        public override void HienThiThongTin()
        {
            Console.WriteLine($"\n--- Thông tin Học viên ---");
            Console.WriteLine($"Mã học viên: {MaHocVien}");
            Console.WriteLine($"Họ và tên: {HoTen}");
            Console.WriteLine($"Email: {Email}");
            Console.WriteLine($"Điểm trung bình: {DiemTongKet:F2}");
            Console.WriteLine($"Trình độ: {TrinhDo}");
            if (CacKhoaHocDaDangKy.Count > 0)
            {
                Console.WriteLine("Khóa học đã đăng ký:");
                foreach (var dangKy in CacKhoaHocDaDangKy)
                {
                    Console.WriteLine($"- {dangKy.KhoaHoc.TenKhoaHoc} (Điểm: {dangKy.DiemSo:F2})");
                }
            }
            else
            {
                Console.WriteLine("Chưa đăng ký khóa học nào.");
            }
            Console.WriteLine("-------------------------");
        }

        private void CapNhatTrinhDo()
        {
            if (CacKhoaHocDaDangKy.Count == 0)
            {
                TrinhDo = "Chưa xếp loại";
                return;
            }
            if (_diemTongKet >= 9)
            {
                TrinhDo = "Xuất sắc";
            }
            else if (_diemTongKet >= 7)
            {
                TrinhDo = "Giỏi";
            }
            else if (_diemTongKet >= 5)
            {
                TrinhDo = "Khá";
            }
            else
            {
                TrinhDo = "Trung bình";
            }
        }

        public void DangKyKhoaHoc(KhoaHoc khoaHoc)
        {
            if (!CacKhoaHocDaDangKy.Any(d => d.KhoaHoc.MaKhoaHoc == khoaHoc.MaKhoaHoc))
            {
                CacKhoaHocDaDangKy.Add(new DangKyKhoaHoc(this, khoaHoc));
                Console.WriteLine($"{HoTen} đã đăng ký khóa học {khoaHoc.TenKhoaHoc}.");
            }
            else
            {
                Console.WriteLine($"{HoTen} đã đăng ký khóa học {khoaHoc.TenKhoaHoc} rồi.");
            }
        }

        public void LamBaiThi(KhoaHoc khoaHoc, double diemSo)
        {
            var dangKy = CacKhoaHocDaDangKy.FirstOrDefault(d => d.KhoaHoc.MaKhoaHoc == khoaHoc.MaKhoaHoc);
            if (dangKy != null)
            {
                if (diemSo < 0 || diemSo > 10)
                {
                    throw new ArgumentOutOfRangeException(nameof(diemSo), "Điểm phải từ 0 đến 10.");
                }
                dangKy.DiemSo = diemSo;

                double tongDiemCacMon = 0;
                if (CacKhoaHocDaDangKy.Any())
                {
                    foreach (var d in CacKhoaHocDaDangKy)
                    {
                        tongDiemCacMon += d.DiemSo;
                    }
                    DiemTongKet = tongDiemCacMon / CacKhoaHocDaDangKy.Count;
                }
                else
                {
                    DiemTongKet = 0;
                }

                Console.WriteLine($"Học viên {HoTen} đã cập nhật điểm {diemSo} cho khóa học {khoaHoc.TenKhoaHoc}.");
            }
            else
            {
                Console.WriteLine($"{HoTen} chưa đăng ký khóa học {khoaHoc.TenKhoaHoc}. Không thể nhập điểm.");
            }
        }

        public void DangNhap()
        {
            Console.WriteLine($"{HoTen} (Mã SV: {MaHocVien}) đã đăng nhập.");
        }

        public void DangXuat()
        {
            Console.WriteLine($"{HoTen} (Mã SV: {MaHocVien}) đã đăng xuất.");
        }
    }
}
