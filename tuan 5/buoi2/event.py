import numpy as np

def find_event_by_id(events, event_id):
    for event in events:
        if event["id"] == event_id:
            return event
    return None

def add_event(events, event_id, name, ticket_price, tickets_left):
    if find_event_by_id(events, event_id):
        print(f"Lỗi: Mã sự kiện '{event_id}' đã tồn tại.")
        return False
    if ticket_price < 0 or tickets_left < 0:
        print("Lỗi: Giá vé và số lượng vé không được âm.")
        return False
    events.append({"id": event_id, "name": name, "ticket_price": ticket_price, "tickets_left": tickets_left})
    print(f"Sự kiện '{name}' đã được thêm thành công.")
    return True

def delete_event(events, event_id):
    event_to_delete = find_event_by_id(events, event_id)
    if event_to_delete:
        events.remove(event_to_delete)
        print(f"Sự kiện có mã '{event_id}' đã được xóa.")
        return True
    else:
        print(f"Lỗi: Không tìm thấy sự kiện có mã '{event_id}'.")
        return False

def update_event_tickets(events, event_id, new_tickets_left):
    event = find_event_by_id(events, event_id)
    if event:
        if new_tickets_left < 0:
            print("Lỗi: Số lượng vé không được âm.")
            return False
        event["tickets_left"] = new_tickets_left
        print(f"Cập nhật số lượng vé cho sự kiện '{event['name']}' thành công.")
        return True
    else:
        print(f"Lỗi: Không tìm thấy sự kiện có mã '{event_id}'.")
        return False

def access_event_info(events, event_id):
    event = find_event_by_id(events, event_id)
    if event:
        print(f"\n--- Thông tin sự kiện ({event_id}) ---")
        for key, value in event.items():
            print(f"{key.replace('_', ' ').capitalize()}: {value}")
    else:
        print(f"Lỗi: Không tìm thấy sự kiện có mã '{event_id}'.")

def print_all_events(events):
    print("\n--- Danh sách tất cả sự kiện ---")
    if not events:
        print("Chưa có sự kiện nào.")
        return
    for event in events:
        print(f"  ID: {event['id']}, Tên: {event['name']}, Giá vé: {event['ticket_price']:.0f} VNĐ, Vé còn lại: {event['tickets_left']}")

def calculate_average_ticket_price(events):
    if not events:
        print("Không có sự kiện nào để tính giá vé trung bình.")
        return 0.0
    ticket_prices = np.array([event["ticket_price"] for event in events])
    average_price = np.mean(ticket_prices)
    print(f"\nGiá vé trung bình của các sự kiện: {average_price:.0f} VNĐ")
    return average_price

def manage_events(events):
    print("\n<<< QUẢN LÝ SỰ KIỆN >>>")
    print_all_events(events)

    print("\nThao tác: Thêm sự kiện mới")
    add_event(events, "EV006", "Lễ hội ẩm thực đường phố", 60000.0, 180)
    add_event(events, "EV001", "Hội chợ sách đã tồn tại", 50000.0, 100)
    add_event(events, "EV007", "Workshop vẽ tranh", -10000.0, 100)

    print("\nThao tác: Truy cập thông tin sự kiện EV002")
    access_event_info(events, "EV002")
    access_event_info(events, "EV999")

    print("\nThao tác: Cập nhật số lượng vé sự kiện EV003")
    update_event_tickets(events, "EV003", 40)
    update_event_tickets(events, "EV003", -5)
    update_event_tickets(events, "EV999", 10)
    access_event_info(events, "EV003")

    print("\nThao tác: Xóa sự kiện EV004")
    delete_event(events, "EV004")
    delete_event(events, "EV999")
    print_all_events(events)

    calculate_average_ticket_price(events)