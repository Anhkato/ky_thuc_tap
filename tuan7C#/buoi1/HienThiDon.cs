using System;

namespace UngDungBanBanh
{
    public static class HienThiDon
    {
        public static void HienThiChiTiet(string tenBanh, int soLuong, double tongTien, string loaiDon)
        {
            Console.WriteLine("\n--- CHI TIẾT ĐƠN ---");
            Console.WriteLine($"Tên bánh: {tenBanh}");
            Console.WriteLine($"Số lượng: {soLuong}");
            Console.WriteLine($"Tổng tiền: {tongTien:N0} VNĐ");
            Console.WriteLine($"Loại đơn: {loaiDon}");
            Console.WriteLine("---------------------");
        }
    }
}