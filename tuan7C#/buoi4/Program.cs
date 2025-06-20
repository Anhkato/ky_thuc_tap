using System;
using System.Collections.Generic;
using System.Linq;

namespace FoodDeliverySystem_VN
{
    class Program
    {
        static void Main(string[] args)
        {
            Console.OutputEncoding = System.Text.Encoding.UTF8;

            var bep = new BoPhanBep();
            var giaoHang = new BoPhanGiaoHang();
            var cskh = new DichVuKhachHang();

            var danhSachDonHang = new List<DonHang>
            {
                new DonHang(101, new List<string> { "Pizza Hải Sản", "Coca-Cola" }),
                new DonHang(102, new List<string> { "Cơm Gà Xối Mỡ" }),
                new DonHang(103, new List<string> { "Bún Bò Huế Tô Đặc Biệt", "Trà Đá" })
            };

            Console.WriteLine("--- ĐĂNG KÝ CÁC OBSERVER VÀO SỰ KIỆN CỦA TỪNG ĐƠN HÀNG ---\n");
            
            foreach (var donHang in danhSachDonHang)
            {
                bep.DangKy(donHang);
                giaoHang.DangKy(donHang);
                cskh.DangKy(donHang);
            }

            Predicate<DonHang> dangDuocGiao = dh => dh.TrangThai == "Đang giao";
            Func<DonHang, string> dinhDangChiTietDonHang = dh => $"Chi tiết: ID={dh.MaDonHang}, Status='{dh.TrangThai}', Items='{string.Join(", ", dh.CacMon)}'";
            Action<string> ghiLogRaConsole = thongBao => Console.WriteLine($"[SYSTEM LOG] {DateTime.Now:HH:mm:ss} - {thongBao}");

            foreach (var donHang in danhSachDonHang)
            {
                donHang.TrangThaiDonHangThayDoi += (sender, e) => {
                    string logMessage = $"Đơn #{e.DonHang.MaDonHang} chuyển từ '{e.TrangThaiCu}' -> '{e.TrangThaiMoi}'";
                    ghiLogRaConsole(logMessage);
                };
            }

            Console.WriteLine("\n--- BẮT ĐẦU MÔ PHỎNG QUÁ TRÌNH XỬ LÝ ĐƠN HÀNG ---\n");

            Console.WriteLine($"\n----- Cập nhật trạng thái cho {danhSachDonHang[0]} -----");
            danhSachDonHang[0].CapNhatTrangThai("Mới tạo");
            danhSachDonHang[0].CapNhatTrangThai("Sẵn sàng giao");
            danhSachDonHang[0].CapNhatTrangThai("Đang giao");
            danhSachDonHang[0].CapNhatTrangThai("Hoàn tất");

            Console.WriteLine($"\n----- Cập nhật trạng thái cho {danhSachDonHang[1]} -----");
            danhSachDonHang[1].CapNhatTrangThai("Mới tạo");
            danhSachDonHang[1].CapNhatTrangThai("Hủy");
            
            Console.WriteLine($"\n----- Cập nhật trạng thái cho {danhSachDonHang[2]} -----");
            danhSachDonHang[2].CapNhatTrangThai("Mới tạo");
            danhSachDonHang[2].CapNhatTrangThai("Sẵn sàng giao");
            danhSachDonHang[2].CapNhatTrangThai("Đang giao");
            danhSachDonHang[2].CapNhatTrangThai("Giao thất bại");

            Console.WriteLine("\n\n--- KIỂM TRA TRẠNG THÁI VỚI PREDICATE VÀ FUNC ---\n");
            foreach (var donHang in danhSachDonHang)
            {
                if (dangDuocGiao(donHang))
                {
                    ghiLogRaConsole($"Đơn hàng sau đây đang trong quá trình vận chuyển: {dinhDangChiTietDonHang(donHang)}");
                }
            }
            
            Console.WriteLine("\n--- THỐNG KÊ (BONUS) ---\n");
            int soDonGiaoThanhCong = danhSachDonHang.Count(dh => dh.TrangThai == "Hoàn tất");
            int soDonDaHuy = danhSachDonHang.Count(dh => dh.TrangThai == "Hủy");

            Console.WriteLine($"Tổng số đơn hàng giao thành công: {soDonGiaoThanhCong}");
            Console.WriteLine($"Tổng số đơn hàng đã bị hủy: {soDonDaHuy}");
        }
    }
}