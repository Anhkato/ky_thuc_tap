
import copy 

from .du_lieu import du_lieu_sach, thong_tin_khach_hang
from .xu_ly_hoa_don import tinh_tien_hoa_don, tim_sach_theo_ten
from .xu_ly_kho import kiem_tra_trang_thai_sach
from .xu_ly_khuyen_mai import tao_ma_giam_gia_cho_khach
from .thong_ke_sach import in_danh_sach_sach_ban_chay, tim_quyen_sach_ban_chay_nhat

def main():
    print("Chào mừng đến với Chương trình Quản lý Cửa hàng Sách!")
    print("="*50)

    danh_sach_sach_goc = du_lieu_sach()
    # Tạo bản sao để không ảnh hương dữ liệu cũ
    danh_sach_sach_hien_tai = copy.deepcopy(danh_sach_sach_goc)
    
    khach_hang_vip, khach_hang_thuong = thong_tin_khach_hang()

    print("\n--- 1. KIỂM TRA TRẠNG THÁI SÁCH ---")
    ten_sach_kt = "Số Đỏ"
    so_luong_kt = 3
    thong_bao_kho, con_hang, phan_loai = kiem_tra_trang_thai_sach(danh_sach_sach_hien_tai, ten_sach_kt, so_luong_kt)
    print(f"Kiểm tra sách '{ten_sach_kt}' với số lượng {so_luong_kt}:")
    print(f"Thông báo: {thong_bao_kho}")
    print(f"Còn hàng: {con_hang}")
    print(f"Phân loại: {phan_loai}")
    
    print("-" * 20)
    ten_sach_kt_2 = "Tắt Đèn" 
    so_luong_kt_2 = 1
    thong_bao_kho_2, con_hang_2, phan_loai_2 = kiem_tra_trang_thai_sach(danh_sach_sach_hien_tai, ten_sach_kt_2, so_luong_kt_2)
    print(f"Kiểm tra sách '{ten_sach_kt_2}' với số lượng {so_luong_kt_2}:")
    print(f"Thông báo: {thong_bao_kho_2}")
    print(f"Còn hàng: {con_hang_2}")
    print(f"Phân loại: {phan_loai_2}")

    print("\n--- 2. TÍNH TOÁN HÓA ĐƠN VÀ MÔ PHỎNG GIAO DỊCH ---")
    ten_sach_mua_vd = "Lão Hạc"
    so_luong_mua_vd = 2
    khach_mua_vd = khach_hang_vip 

    _, con_hang_de_mua, _ = kiem_tra_trang_thai_sach(danh_sach_sach_hien_tai, ten_sach_mua_vd, so_luong_mua_vd)

    if con_hang_de_mua:
        thong_bao_hd, tong_tien_hd = tinh_tien_hoa_don(danh_sach_sach_hien_tai, ten_sach_mua_vd, so_luong_mua_vd, khach_mua_vd)
        print(thong_bao_hd)

        if tong_tien_hd is not None: # Nếu tính toán thành công (không phải thông báo lỗi)
            print(f"Khách hàng: {khach_mua_vd['ten_khach_hang']} ({khach_mua_vd['loai_khach_hang']})")
            # Mô phỏng cập nhật kho sau khi bán
            sach_da_ban = tim_sach_theo_ten(danh_sach_sach_hien_tai, ten_sach_mua_vd)
            if sach_da_ban:
                sach_da_ban["so_luong_ton_kho"] -= so_luong_mua_vd
                sach_da_ban["so_luong_da_ban"] += so_luong_mua_vd
                print(f"Đã cập nhật kho cho sách '{ten_sach_mua_vd}': Tồn kho mới {sach_da_ban['so_luong_ton_kho']}, Đã bán mới {sach_da_ban['so_luong_da_ban']}")
    else:
        print(f"Không thể thực hiện giao dịch cho sách '{ten_sach_mua_vd}' do không đủ hàng.")

    print("-" * 20)
    # mua sách không đủ hàng
    ten_sach_mua_vd_2 = "Số Đỏ"
    so_luong_mua_vd_2 = 10 
    khach_mua_vd_2 = khach_hang_thuong
    
    thong_bao_hd_2, tong_tien_hd_2 = tinh_tien_hoa_don(danh_sach_sach_hien_tai, ten_sach_mua_vd_2, so_luong_mua_vd_2, khach_mua_vd_2)
    print(thong_bao_hd_2) 

    print("\n--- 3. TẠO MÃ GIẢM GIÁ ---")
    ma_giam_gia_vip = tao_ma_giam_gia_cho_khach(khach_hang_vip["ten_khach_hang"], khach_hang_vip["loai_khach_hang"])
    print(f"Mã giảm giá cho khách {khach_hang_vip['ten_khach_hang']} (VIP): {ma_giam_gia_vip}")

    ma_giam_gia_thuong = tao_ma_giam_gia_cho_khach(khach_hang_thuong["ten_khach_hang"], khach_hang_thuong["loai_khach_hang"])
    print(f"Mã giảm giá cho khách {khach_hang_thuong['ten_khach_hang']} (thường): {ma_giam_gia_thuong}")

    print("\n--- 4. THỐNG KÊ SÁCH ---")
    # Danh sách sách hiện tại (đã cập nhật sau giao dịch)
    in_danh_sach_sach_ban_chay(danh_sach_sach_hien_tai, 10) 
    tim_quyen_sach_ban_chay_nhat(danh_sach_sach_hien_tai)
    
    print("\n" + "="*50)
    print("Chương trình kết thúc.")

if __name__ == "__main__":
    main()