def tim_sach_theo_ten(danh_sach_sach, ten_sach_can_tim):
    for sach in danh_sach_sach:
        if sach["ten_sach"].lower() == ten_sach_can_tim.lower():
            return sach
    return None

def tinh_tien_hoa_don(danh_sach_sach, ten_sach_mua, so_luong_mua, khach_hang):
    if not isinstance(so_luong_mua, int) or so_luong_mua <= 0:
        return "Số lượng mua phải là số nguyên dương.", None

    sach_can_mua = tim_sach_theo_ten(danh_sach_sach, ten_sach_mua)

    if sach_can_mua is None:
        return f"Không tìm thấy sách '{ten_sach_mua}'.", None

    if sach_can_mua["so_luong_ton_kho"] == 0:
        return f"Sách '{ten_sach_mua}' đã hết hàng.", None

    if so_luong_mua > sach_can_mua["so_luong_ton_kho"]:
        return f"Không đủ số lượng sách '{ten_sach_mua}' trong kho (còn {sach_can_mua['so_luong_ton_kho']}).", None

    gia_sach = sach_can_mua["gia"]
    tong_tien = gia_sach * so_luong_mua

    loai_khach_hang = khach_hang.get("loai_khach_hang", "thường")

    if loai_khach_hang == "VIP":
        tong_tien *= 0.9  # Giảm 10%

    return f"Tổng tiền hóa đơn cho {so_luong_mua} cuốn '{ten_sach_mua}': {tong_tien:,.0f} VNĐ", float(tong_tien)