def in_danh_sach_sach_ban_chay(danh_sach_sach, nguong_ban_chay=10):
    print(f"\n--- DANH SÁCH CÁC CUỐN SÁCH BÁN CHẠY (TRÊN {nguong_ban_chay} CUỐN) ---")
    co_sach_ban_chay = False
    for sach in danh_sach_sach:
        if sach["so_luong_da_ban"] > nguong_ban_chay:
            print(f"- Tên sách: {sach['ten_sach']}, Số lượng đã bán: {sach['so_luong_da_ban']}")
            co_sach_ban_chay = True
    if not co_sach_ban_chay:
        print(f"Không có sách nào bán được trên {nguong_ban_chay} cuốn.")

def tim_quyen_sach_ban_chay_nhat(danh_sach_sach):
    print("\n--- TÌM CUỐN SÁCH BÁN CHẠY NHẤT ---")
    if not danh_sach_sach:
        print("Danh sách sách trống.")
        return

    sach_ban_chay_nhat = None
    max_so_luong_ban = -1
    

    i = 0
    while i < len(danh_sach_sach):
        sach_hien_tai = danh_sach_sach[i]
        if sach_hien_tai["so_luong_da_ban"] > max_so_luong_ban:
            max_so_luong_ban = sach_hien_tai["so_luong_da_ban"]
            sach_ban_chay_nhat = sach_hien_tai
        i += 1

    if sach_ban_chay_nhat:
        print(f"Sách bán chạy nhất là: '{sach_ban_chay_nhat['ten_sach']}' với {sach_ban_chay_nhat['so_luong_da_ban']} cuốn đã bán.")
    else:
        print("Không có dữ liệu về sách đã bán.")