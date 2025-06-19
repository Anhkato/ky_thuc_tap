using System;

namespace UngDungBanBanh
{
    class MainProgram
    {
        static void Main(string[] args)
        {
            QuanLyDon.ChaoVaHuongDan();

            QuanLyDon.HienThiBanhCoSan();

            int tongSoDon = 0;
            double tongDoanhThu = 0;
            int donLonCount = 0;
            int donThuongCount = 0;

            while (true)
            {
                Console.WriteLine("\n--- NHẬP ĐƠN HÀNG MỚI ---");
                Console.Write("Nhập tên bánh muốn mua (hoặc 'thoat' để dừng): ");
                string? tenBanhNhapGoc = Console.ReadLine();

                if (tenBanhNhapGoc?.ToLower() == "thoat")
                {
                    break;
                }

                string tenBanhChuanHoa = tenBanhNhapGoc!.ToLower();

                if (!DuLieuBanh.GiaBanh.ContainsKey(tenBanhChuanHoa))
                {
                    Console.WriteLine("Loại bánh này không tồn tại. Vui lòng kiểm tra lại tên bánh hoặc gõ 'thoat' để xem lại danh sách.");
                    continue;
                }

                Console.Write("Nhập số lượng: ");
                string? soLuongNhap = Console.ReadLine();
                int soLuong;

                if (!int.TryParse(soLuongNhap, out soLuong) || soLuong <= 0)
                {
                    Console.WriteLine("Số lượng không hợp lệ. Vui lòng nhập số nguyên dương.");
                    continue;
                }

                double tongTienDon = TinhToanDon.TinhTien(tenBanhChuanHoa, soLuong);

                string loaiDon = TinhToanDon.PhanLoaiDon(tongTienDon);

                HienThiDon.HienThiChiTiet(tenBanhNhapGoc!, soLuong, tongTienDon, loaiDon);

                tongSoDon++;
                tongDoanhThu += tongTienDon;
                if (loaiDon == "Đơn lớn")
                {
                    donLonCount++;
                }
                else
                {
                    donThuongCount++;
                }
            }

            QuanLyDon.HienThiThongKeNgay(tongSoDon, tongDoanhThu, donLonCount, donThuongCount);

            Console.WriteLine("\nCảm ơn bạn đã dùng ứng dụng!");
            Console.ReadKey();
        }
    }
}