using HeThongQuanLyHocVien.Models;
using HeThongQuanLyHocVien.Services;
using System;
using System.Collections.Generic;
using System.Linq;

namespace HeThongQuanLyHocVien
{
    class Program
    {
        static DichVuHocVien _dichVuHocVien = new DichVuHocVien();

        static void Main(string[] args)
        {
            Console.OutputEncoding = System.Text.Encoding.UTF8;
            ChayMenu();
        }

        static void ChayMenu()
        {
            while (true)
            {
                HienThiMenu();
                Console.Write("Chọn chức năng: ");
                string? luaChon = Console.ReadLine();

                switch (luaChon)
                {
                    case "1":
                        ThemHocVienHandler();
                        break;
                    case "2":
                        DangKyKhoaHocHandler();
                        break;
                    case "3":
                        NhapDiemHandler();
                        break;
                    case "4":
                        _dichVuHocVien.HienThiTatCaHocVien();
                        break;
                    case "5":
                        _dichVuHocVien.LuuDuLieu();
                        break;
                    case "6":
                        Console.WriteLine("Đang thoát chương trình. Tạm biệt!");
                        return;
                    default:
                        Console.WriteLine("Lựa chọn không hợp lệ. Vui lòng thử lại.");
                        break;
                }
                Console.WriteLine("\nNhấn phím bất kỳ để tiếp tục...");
                Console.ReadKey();
                Console.Clear();
            }
        }

        static void HienThiMenu()
        {
            Console.WriteLine("--- HỆ THỐNG QUẢN LÝ HỌC VIÊN ---");
            Console.WriteLine("1. Thêm học viên mới");
            Console.WriteLine("2. Đăng ký khóa học cho học viên");
            Console.WriteLine("3. Nhập điểm cho học viên");
            Console.WriteLine("4. Hiển thị danh sách học viên");
            Console.WriteLine("5. Ghi dữ liệu học viên ra file");
            Console.WriteLine("6. Thoát");
            Console.WriteLine("---------------------------------");
        }

        static void ThemHocVienHandler()
        {
            Console.WriteLine("\n--- THÊM HỌC VIÊN MỚI ---");
            Console.Write("Nhập họ: ");
            string? ho = Console.ReadLine();
            Console.Write("Nhập tên: ");
            string? ten = Console.ReadLine();
            Console.Write("Nhập Email: ");
            string? email = Console.ReadLine();

            if (string.IsNullOrWhiteSpace(ho) || string.IsNullOrWhiteSpace(ten) || string.IsNullOrWhiteSpace(email))
            {
                Console.WriteLine("Họ, tên và email không được để trống.");
                return;
            }

            _dichVuHocVien.ThemHocVien(ho!, ten!, email!);
        }

        static void DangKyKhoaHocHandler()
        {
            Console.WriteLine("\n--- ĐĂNG KÝ KHÓA HỌC ---");
            _dichVuHocVien.HienThiTatCaHocVien();
            _dichVuHocVien.HienThiTatCaKhoaHoc();

            Console.Write("Nhập Mã học viên: ");
            if (!int.TryParse(Console.ReadLine(), out int maHocVien))
            {
                Console.WriteLine("Lỗi: Mã học viên không hợp lệ.");
                return;
            }

            Console.Write("Nhập Mã khóa học: ");
            if (!int.TryParse(Console.ReadLine(), out int maKhoaHoc))
            {
                Console.WriteLine("Lỗi: Mã khóa học không hợp lệ.");
                return;
            }

            _dichVuHocVien.DangKyKhoaHocChoHocVien(maHocVien, maKhoaHoc);
        }

        static void NhapDiemHandler()
        {
            Console.WriteLine("\n--- NHẬP ĐIỂM HỌC VIÊN ---");
            _dichVuHocVien.HienThiTatCaHocVien(); 

            Console.Write("Nhập Mã học viên: ");
            if (!int.TryParse(Console.ReadLine(), out int maHocVien))
            {
                Console.WriteLine("Lỗi: Mã học viên không hợp lệ.");
                return;
            }

            // Lấy danh sách khóa học mà học viên đã đăng ký
            List<KhoaHoc>? cacKhoaHocDaDangKy = _dichVuHocVien.LayKhoaHocDaDangKyCuaHocVien(maHocVien);

            if (cacKhoaHocDaDangKy == null)
            {
                Console.WriteLine($"Lỗi: Không tìm thấy học viên với mã {maHocVien}.");
                return;
            }
            if (cacKhoaHocDaDangKy.Count == 0)
            {
                Console.WriteLine($"Học viên {maHocVien} chưa đăng ký khóa học nào. Vui lòng đăng ký trước.");
                return;
            }

            Console.WriteLine($"\n--- KHÓA HỌC ĐÃ ĐĂNG KÝ CỦA HỌC VIÊN {maHocVien} ---");
            foreach (var kh in cacKhoaHocDaDangKy)
            {
                kh.HienThiThongTinKhoaHoc();
            }
            Console.WriteLine("------------------------------------");

            Console.Write("Nhập Mã khóa học cần nhập điểm: ");
            if (!int.TryParse(Console.ReadLine(), out int maKhoaHoc))
            {
                Console.WriteLine("Lỗi: Mã khóa học không hợp lệ.");
                return;
            }

            Console.Write("Nhập điểm (0-10): ");
            if (!double.TryParse(Console.ReadLine(), out double diemSo))
            {
                Console.WriteLine("Lỗi: Điểm không hợp lệ.");
                return;
            }

            _dichVuHocVien.NhapDiemHocVien(maHocVien, maKhoaHoc, diemSo);
        }
    }
}
