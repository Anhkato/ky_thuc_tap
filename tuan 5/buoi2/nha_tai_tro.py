def find_sponsor_by_id(sponsors, sponsor_id):
    return sponsors.get(sponsor_id)

def add_sponsor(sponsors, sponsor_id, name, amount):
    if sponsor_id in sponsors:
        print(f"Lỗi: Mã nhà tài trợ '{sponsor_id}' đã tồn tại.")
        return False
    if amount < 0:
        print("Lỗi: Số tiền tài trợ không được âm.")
        return False
    sponsors[sponsor_id] = (name, amount)
    print(f"Nhà tài trợ '{name}' đã được thêm thành công.")
    return True

def delete_sponsor(sponsors, sponsor_id):
    if sponsor_id in sponsors:
        del sponsors[sponsor_id]
        print(f"Nhà tài trợ có mã '{sponsor_id}' đã được xóa.")
        return True
    else:
        print(f"Lỗi: Không tìm thấy nhà tài trợ có mã '{sponsor_id}'.")
        return False

def update_sponsor_amount(sponsors, sponsor_id, new_amount):
    if sponsor_id in sponsors:
        if new_amount < 0:
            print("Lỗi: Số tiền tài trợ không được âm.")
            return False
        name, _ = sponsors[sponsor_id]
        sponsors[sponsor_id] = (name, new_amount)
        print(f"Cập nhật số tiền tài trợ cho '{name}' thành công.")
        return True
    else:
        print(f"Lỗi: Không tìm thấy nhà tài trợ có mã '{sponsor_id}'.")
        return False

def access_sponsor_info(sponsors, sponsor_id):
    sponsor_info = find_sponsor_by_id(sponsors, sponsor_id)
    if sponsor_info:
        name, amount = sponsor_info
        print(f"\n--- Thông tin nhà tài trợ ({sponsor_id}) ---")
        print(f"Tên nhà tài trợ: {name}")
        print(f"Số tiền tài trợ: {amount:.0f} VNĐ")
    else:
        print(f"Lỗi: Không tìm thấy nhà tài trợ có mã '{sponsor_id}'.")

def print_all_sponsors(sponsors):
    print("\n--- Danh sách tất cả nhà tài trợ ---")
    if not sponsors:
        print("Chưa có nhà tài trợ nào.")
        return
    for sponsor_id, (name, amount) in sponsors.items():
        print(f"  ID: {sponsor_id}, Tên: {name}, Số tiền: {amount:.0f} VNĐ")

def manage_sponsors(sponsors):
    print("\n<<< QUẢN LÝ NHÀ TÀI TRỢ >>>")
    print_all_sponsors(sponsors)

    print("\nThao tác: Thêm nhà tài trợ mới")
    add_sponsor(sponsors, "SP004", "Ngân hàng Phát Triển Cộng Đồng", 200000000.0)
    add_sponsor(sponsors, "SP001", "Công ty A đã tồn tại", 10000000.0)
    add_sponsor(sponsors, "SP005", "Nhà hảo tâm ẩn danh", -5000000.0)

    print("\nThao tác: Truy cập thông tin nhà tài trợ SP002")
    access_sponsor_info(sponsors, "SP002")
    access_sponsor_info(sponsors, "SP999")

    print("\nThao tác: Cập nhật số tiền tài trợ cho SP003")
    update_sponsor_amount(sponsors, "SP003", 85000000.0)
    update_sponsor_amount(sponsors, "SP003", -1000000.0)
    update_sponsor_amount(sponsors, "SP999", 10000000.0)
    access_sponsor_info(sponsors, "SP003")

    print("\nThao tác: Xóa nhà tài trợ SP001")
    delete_sponsor(sponsors, "SP001")
    delete_sponsor(sponsors, "SP999")
    print_all_sponsors(sponsors)