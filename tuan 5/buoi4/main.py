import pendulum 

from utils import (
    validate_input, 
    calculate_cost, 
    save_registration, 
    load_registrations
)
from config import COURSES_DB

def main():
  
    print("--- CHƯƠG TRÌNH QUẢN LÝ ĐĂNG KÝ KHÓA HỌC (MODULAR) ---")

    while True:
        full_name = input("Nhập họ và tên: ").strip()
        if len(full_name) >= 2:
            break
        print("Lỗi: Họ tên phải có ít nhất 2 ký tự. Vui lòng thử lại.")
        
    email = validate_input(
        prompt="Nhập email: ",
        regex_pattern=r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$",
        error_message="Định dạng email không hợp lệ (ví dụ: ten@example.com)"
    )

    print("\nDanh sách các khóa học hiện có:")
    for code, details in COURSES_DB.items():
        print(f"- {code}: {details['name']} ({details['price']:,.0f} VNĐ)")

    course_id = validate_input(
        prompt="Nhập mã khóa học bạn muốn đăng ký: ",
        regex_pattern=r"^KH\d{3}$",
        error_message="Định dạng mã không hợp lệ (ví dụ: KH001)",
        ignore_case=True  
    )
    
    if course_id not in COURSES_DB:
        print(f"Lỗi: Mã khóa học '{course_id}' không có trong danh sách. Vui lòng chạy lại chương trình.")
        return

    discount_code = input("Nhập mã ưu đãi (nếu có, ví dụ: SUMMER25): ").strip()
    cost = calculate_cost(course_id, discount_code)
    
    if cost is None:
        return
        
    print(f"\n Tổng chi phí cho khóa học '{COURSES_DB[course_id]['name']}' là: {cost:,.0f} VNĐ.")

    registration_datetime = pendulum.now('Asia/Ho_Chi_Minh')
    registration_date_str = registration_datetime.format('DD/MM/YYYY HH:mm:ss')

    confirmation_message = (
        f" Chúc mừng {full_name} đã đăng ký thành công khóa học {course_id} "
        f"vào ngày {registration_date_str}!"
    )
    print(confirmation_message)

    registration_info = {
        "ho_ten": full_name,
        "email": email,
        "ma_khoa_hoc": course_id,
        "ngay_dang_ky": registration_date_str,
        "chi_phi": cost
    }
    save_registration(registration_info)

    load_registrations()


if __name__ == "__main__":
    main()