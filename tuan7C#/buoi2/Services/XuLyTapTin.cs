using HeThongQuanLyHocVien.Models;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace HeThongQuanLyHocVien.Services
{
    public static class XuLyTapTin
    {
        private static readonly string _duongDanTapTin = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "Data", "hocvien.csv");

        public static void LuuHocVienVaoTapTin(List<HocVien> danhSachHocVien)
        {
            try
            {
                Directory.CreateDirectory(Path.GetDirectoryName(_duongDanTapTin)!);

                using (StreamWriter ghi = new StreamWriter(_duongDanTapTin))
                {
                    ghi.WriteLine("MaHocVien,Ho,Ten,Email,DiemTongKet,TrinhDo,CacKhoaHocDaDangKy");

                    foreach (var hocVien in danhSachHocVien)
                    {
                        string chuoiDangKy = string.Join(";", hocVien.CacKhoaHocDaDangKy.Select(d => $"{d.KhoaHoc.MaKhoaHoc}-{d.KhoaHoc.TenKhoaHoc}-{d.DiemSo:F2}"));
                        ghi.WriteLine($"{hocVien.MaHocVien},{hocVien.Ho},{hocVien.Ten},{hocVien.Email},{hocVien.DiemTongKet:F2},{hocVien.TrinhDo},{chuoiDangKy}");
                    }
                }
                Console.WriteLine("Đã lưu dữ liệu học viên vào tập tin thành công.");
            }
            catch (IOException ex)
            {
                Console.WriteLine($"Lỗi IO khi ghi tập tin: {ex.Message}");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Lỗi không xác định khi ghi tập tin: {ex.Message}");
            }
        }

        public static List<HocVien> TaiHocVienTuTapTin(List<KhoaHoc> danhSachKhoaHocHienCo)
        {
            List<HocVien> danhSachHocVien = new List<HocVien>();
            if (!File.Exists(_duongDanTapTin))
            {
                Console.WriteLine("Tập tin dữ liệu học viên không tồn tại. Bắt đầu với danh sách trống.");
                return danhSachHocVien;
            }

            try
            {
                using (StreamReader doc = new StreamReader(_duongDanTapTin))
                {
                    doc.ReadLine();

                    string? dong;
                    while ((dong = doc.ReadLine()) != null)
                    {
                        string[] cacPhan = dong.Split(',');
                        if (cacPhan.Length >= 6)
                        {
                            int maHocVien = int.Parse(cacPhan[0]);
                            string ho = cacPhan[1];
                            string ten = cacPhan[2];
                            string email = cacPhan[3];
                            double diemTongKet = double.Parse(cacPhan[4]);

                            HocVien hocVien = new HocVien(maHocVien, ho, ten, email);
                            hocVien.DiemTongKet = diemTongKet;

                            if (cacPhan.Length > 6 && !string.IsNullOrWhiteSpace(cacPhan[6]))
                            {
                                string duLieuDangKy = cacPhan[6];
                                string[] tungDangKy = duLieuDangKy.Split(';');
                                foreach (var chuoiDangKy in tungDangKy)
                                {
                                    if (string.IsNullOrWhiteSpace(chuoiDangKy)) continue;

                                    string[] phanDangKy = chuoiDangKy.Split('-');
                                    if (phanDangKy.Length == 3)
                                    {
                                        int maKhoaHoc = int.Parse(phanDangKy[0]);
                                        double diemKhoaHoc = double.Parse(phanDangKy[2]);

                                        var khoaHoc = danhSachKhoaHocHienCo.FirstOrDefault(k => k.MaKhoaHoc == maKhoaHoc);
                                        if (khoaHoc != null)
                                        {
                                            hocVien.CacKhoaHocDaDangKy.Add(new DangKyKhoaHoc(hocVien, khoaHoc) { DiemSo = diemKhoaHoc });
                                        }
                                    }
                                }
                            }
                            danhSachHocVien.Add(hocVien);
                        }
                    }
                }
                Console.WriteLine("Đã đọc dữ liệu học viên từ tập tin thành công.");
            }
            catch (FileNotFoundException)
            {
                Console.WriteLine("Lỗi: Tập tin dữ liệu học viên không tồn tại.");
            }
            catch (FormatException ex)
            {
                Console.WriteLine($"Lỗi định dạng dữ liệu trong tập tin: {ex.Message}. Vui lòng kiểm tra tập tin.");
            }
            catch (IOException ex)
            {
                Console.WriteLine($"Lỗi IO khi đọc tập tin: {ex.Message}");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Lỗi không xác định khi đọc tập tin: {ex.Message}");
            }
            return danhSachHocVien;
        }
    }
}
