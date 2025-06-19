using System;

namespace UngDungBanBanh
{
    public static class QuanLyDon
    {
        public static void ChaoVaHuongDan()
        {
            Console.WriteLine("*************************************");
            Console.WriteLine("* CHÀO MỪNG ĐẾN VỚI TIỆM BÁNH!    *");
            Console.WriteLine("*************************************");
            Console.WriteLine("\nHướng dẫn:");
            Console.WriteLine("- Nhập tên bánh và số lượng.");
            Console.WriteLine("- Gõ 'thoat' để dừng.");
            Console.WriteLine("- Hệ thống sẽ tính tiền và phân loại đơn.");
            Console.WriteLine("- Cuối ngày có thống kê doanh thu.");
        }

        public static void HienThiBanhCoSan()
        {
            Console.WriteLine("\n--- BÁNH CÓ SẴN ---");
            int stt = 1;
            foreach (var banh in DuLieuBanh.GiaBanh)
            {
                Console.WriteLine($"{stt}. {banh.Key}: {banh.Value:N0} VNĐ");
                stt++;
            }
        }

        public static void HienThiThongKeNgay(int tongDon, double tongDoanhThu, int donLon, int donThuong)
        {
            Console.WriteLine("\n--- THỐNG KÊ NGÀY ---");
            Console.WriteLine($"Tổng số đơn: {tongDon}");
            Console.WriteLine($"Tổng doanh thu: {tongDoanhThu:N0} VNĐ");
            Console.WriteLine($"Đơn lớn (> 100.000 VNĐ): {donLon}");
            Console.WriteLine($"Đơn thường (<= 100.000 VNĐ): {donThuong}");
        }
    }
}