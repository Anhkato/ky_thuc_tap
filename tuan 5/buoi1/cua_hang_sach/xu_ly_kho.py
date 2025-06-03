
def tim_sach_theo_ten_trong_kho(danh_sach_sach, ten_sach_can_tim): 
    for sach in danh_sach_sach:
        if sach["ten_sach"].lower() == ten_sach_can_tim.lower():
            return sach
    return None

def kiem_tra_trang_thai_sach(danh_sach_sach, ten_sach_can_kiem_tra, so_luong_muon_mua):
    sach = tim_sach_theo_ten_trong_kho(danh_sach_sach, ten_sach_can_kiem_tra)
    thong_bao_ton_kho = ""
    trang_thai_ton_kho = False
    phan_loai_gia = "Không xác định"

    if sach is None:
        thong_bao_ton_kho = f"Không tìm thấy sách '{ten_sach_can_kiem_tra}'."
        return thong_bao_ton_kho, trang_thai_ton_kho, phan_loai_gia

    if not isinstance(so_luong_muon_mua, int) or so_luong_muon_mua <= 0:
        thong_bao_ton_kho = "Số lượng muốn mua phải là số nguyên dương."
       
    elif sach["so_luong_ton_kho"] >= so_luong_muon_mua:
        thong_bao_ton_kho = f"Sách '{ten_sach_can_kiem_tra}' còn hàng (đủ {so_luong_muon_mua} cuốn)."
        trang_thai_ton_kho = True
    else:
        thong_bao_ton_kho = f"Sách '{ten_sach_can_kiem_tra}' hết hàng hoặc không đủ số lượng yêu cầu (còn {sach['so_luong_ton_kho']})."
        trang_thai_ton_kho = False

    gia_sach = sach["gia"]
    match gia_sach:
        case gia if gia < 50000:
            phan_loai_gia = "Sách giá rẻ"
        case gia if 50000 <= gia <= 100000:
            phan_loai_gia = "Sách giá tầm trung"
        case gia if gia > 100000:
            phan_loai_gia = "Sách cao cấp"
        case _:
            phan_loai_gia = "Không phân loại giá"
            
    return thong_bao_ton_kho, trang_thai_ton_kho, phan_loai_gia