import quan_ly_nhat_ky as ql

def in_menu():
#    menu
    print("\n--- CÔNG CỤ QUẢN LÝ NHẬT KÝ TUẦN ---")
    print("1. Tạo nhật ký tuần mới")
    print("2. Đọc nhật ký tuần")
    print("3. Cập nhật nhật ký tuần")
    print("4. Xóa nhật ký tuần")
    print("5. Xem báo cáo tổng kết")
    print("6. Thoát")
    print("-------------------------------------")

def xu_ly_luu(is_update=False):
    # tạo mới và cập nhật
    hanh_dong = "cập nhật" if is_update else "tạo"
    try:
        so_tuan = int(input(f"Nhập số tuần cần {hanh_dong}: "))
        
        if not is_update and ql.doc_nhat_ky(so_tuan) is not None:
            if input(f"Tuần {so_tuan} đã tồn tại. Ghi đè? (c/k): ").lower() != 'c':
                print("Đã hủy.")
                return

        so_gio = float(input("Nhập số giờ làm việc: "))
        so_nhiem_vu = int(input("Nhập số nhiệm vụ hoàn thành: "))
        ghi_chu = input("Nhập ghi chú: ")

        if ql.luu_nhat_ky(so_tuan, so_gio, so_nhiem_vu, ghi_chu):
            print(f" Đã {hanh_dong} nhật ký tuần {so_tuan} thành công!")
        else:
            print(f" Không thể {hanh_dong} nhật ký.")
            
    except ValueError:
        print(" Lỗi: Dữ liệu nhập không hợp lệ.")

def xu_ly_doc():
    try:
        so_tuan = int(input("Nhập số tuần cần đọc: "))
        noi_dung = ql.doc_nhat_ky(so_tuan)

        if noi_dung and noi_dung.strip():
            print(f"\n--- Nội dung nhật ký tuần {so_tuan} ---")
            print(noi_dung)
        else:
            print(f"Không tìm thấy nhật ký hoặc nhật ký tuần {so_tuan} không có nội dung.")

    except ValueError:
        print(" Lỗi: Số tuần phải là một số nguyên.")
    input("\nNhấn Enter để quay về menu...")

def xu_ly_xoa():
    # xoa nhat ky
    try:
        so_tuan = int(input("Nhập số tuần cần xóa: "))
        if ql.xoa_nhat_ky(so_tuan):
            print(f" Đã xóa nhật ký tuần {so_tuan}.")
        else:
            print(f" Không tìm thấy nhật ký tuần {so_tuan}.")
    except ValueError:
        print(" Lỗi: Số tuần phải là một số nguyên.")

def xu_ly_bao_cao():
    """ tạo và hiển thị báo cáo."""
    bao_cao = ql.tao_bao_cao()
    print("\n--- BÁO CÁO TỔNG KẾT ---")
    if bao_cao["tuan"] == 0:
        print("Chưa có dữ liệu nào.")
    else:
        print(f"Tổng số tuần đã ghi nhận: {bao_cao['tuan']}")
        print(f"Tổng số giờ làm việc: {bao_cao['gio']:.2f}")
        print(f"Tổng nhiệm vụ hoàn thành: {bao_cao['nhiem_vu']}")
    print("--------------------------")

def main():
    chuc_nang = {
        '1': lambda: xu_ly_luu(is_update=False),
        '2': xu_ly_doc,
        '3': lambda: xu_ly_luu(is_update=True),
        '4': xu_ly_xoa,
        '5': xu_ly_bao_cao,
    }
    
    while True:
        in_menu()
        lua_chon = input("Vui lòng chọn một chức năng (1-6): ")

        if lua_chon == '6':
            print(" Tạm biệt!")
            break
        
        # Lấy hàm từ dict và thực thi
        ham_xu_ly = chuc_nang.get(lua_chon)
        if ham_xu_ly:
            ham_xu_ly()
        else:
            print(" Lựa chọn không hợp lệ. Vui lòng chọn lại.")

if __name__ == "__main__":
    main()