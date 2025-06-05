import json
import re

from config import COURSES_DB, DISCOUNT_CODES, JSON_FILE

def validate_input(prompt, regex_pattern, error_message, ignore_case=False):
   
    while True:
        user_input = input(prompt).strip()
        
        flags = re.IGNORECASE if ignore_case else 0
        
        if re.match(regex_pattern, user_input, flags):
            if ignore_case:
                return user_input.upper()
            return user_input
        else:
            print(f"Lỗi: {error_message}. Vui lòng thử lại.")

def calculate_cost(course_id, discount_code):
   
    try:
        base_price = COURSES_DB[course_id]["price"]
        discount_percentage = DISCOUNT_CODES.get(discount_code.upper(), 0)
        final_cost = base_price * (1 - discount_percentage)
        return round(final_cost, 2)
    except KeyError:
        print(f"Lỗi: Mã khóa học '{course_id}' không tồn tại.")
        return None

def save_registration(registration_data):
    try:
        with open(JSON_FILE, 'r', encoding='utf-8') as f:
            registrations = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        registrations = []

    registrations.append(registration_data)

    try:
        with open(JSON_FILE, 'w', encoding='utf-8') as f:
            json.dump(registrations, f, indent=4, ensure_ascii=False)
        print("Đã lưu thông tin đăng ký thành công!")
    except IOError as e:
        print(f"Lỗi: Không thể ghi dữ liệu vào tệp. Chi tiết: {e}")

def load_registrations():
    print("\n--- DANH SÁCH HỌC VIÊN ĐÃ ĐĂNG KÝ ---")
    try:
        with open(JSON_FILE, 'r', encoding='utf-8') as f:
            registrations = json.load(f)
        
        if not registrations:
            print("Chưa có học viên nào đăng ký.")
            return

        for reg in registrations:
            name = reg.get("ho_ten", "N/A")
            course_id = reg.get("ma_khoa_hoc", "N/A")
            reg_date = reg.get("ngay_dang_ky", "N/A")
            cost = reg.get("chi_phi", 0)
            
            print(
                f" Học viên: {name} | "
                f" Khóa học: {course_id} | "
                f" Ngày: {reg_date} | "
                f" Chi phí: {cost:,.0f} VNĐ"
            )

    except FileNotFoundError:
        print("Chưa có dữ liệu đăng ký nào được lưu.")
    except json.JSONDecodeError:
        print(f"Lỗi: Tệp '{JSON_FILE}' có định dạng không hợp lệ.")
    except Exception as e:
        print(f"Đã xảy ra lỗi không mong muốn khi đọc tệp: {e}")