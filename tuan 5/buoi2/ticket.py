from event import find_event_by_id

_ticket_counter = 0

def _generate_ticket_id():
    global _ticket_counter
    _ticket_counter += 1
    return f"TICKET_{_ticket_counter:03d}"

def add_ticket_sale(events, ticket_sales_history, sold_events_today, event_id, quantity):
    event = find_event_by_id(events, event_id)
    if not event:
        print(f"Lỗi: Không tìm thấy sự kiện có mã '{event_id}' để bán vé.")
        return False

    if quantity <= 0:
        print("Lỗi: Số lượng vé bán phải lớn hơn 0.")
        return False

    if event["tickets_left"] < quantity:
        print(f"Lỗi: Sự kiện '{event['name']}' không đủ vé (còn {event['tickets_left']}, cần {quantity}).")
        return False

    ticket_id = _generate_ticket_id()
    sale_transaction = {
        "event_id": event_id,
        "ticket_id": ticket_id,
        "quantity": quantity
    }
    ticket_sales_history.append(sale_transaction)
    event["tickets_left"] -= quantity
    sold_events_today.add(event_id)
    print(f"Bán {quantity} vé (Mã vé: {ticket_id}) cho sự kiện '{event['name']}' thành công. Vé còn lại: {event['tickets_left']}")
    return True

def check_event_sold_today(sold_events_today, event_id):
    if event_id in sold_events_today:
        print(f"Sự kiện có mã '{event_id}' CÓ vé bán trong ngày.")
        return True
    else:
        print(f"Sự kiện có mã '{event_id}' KHÔNG CÓ vé bán trong ngày.")
        return False

def print_sales_by_event(ticket_sales_history, event_id_filter=None):
    print("\n--- Lịch sử giao dịch bán vé ---")
    if not ticket_sales_history:
        print("Chưa có giao dịch bán vé nào.")
        return

    filtered_sales = [
        sale for sale in ticket_sales_history
        if event_id_filter is None or sale["event_id"] == event_id_filter
    ]

    if not filtered_sales:
        if event_id_filter:
            print(f"Không có giao dịch nào cho sự kiện '{event_id_filter}'.")
        else:
            print("Không có giao dịch nào.")
        return

    if event_id_filter:
        print(f"Giao dịch cho sự kiện: {event_id_filter}")

    for sale in filtered_sales:
        print(f"  Sự kiện ID: {sale['event_id']}, Mã vé: {sale['ticket_id']}, Số lượng: {sale['quantity']}")

def remove_zero_quantity_sales(ticket_sales_history):
    initial_count = len(ticket_sales_history)
    ticket_sales_history[:] = [sale for sale in ticket_sales_history if sale["quantity"] > 0]
    removed_count = initial_count - len(ticket_sales_history)
    if removed_count > 0:
        print("Đã xóa {removed_count} giao dịch với số lượng vé bằng 0.")
    else:
        print("Không có giao dịch nào với số lượng vé bằng 0 để xóa.")

def process_tickets(events, ticket_sales_history, sold_events_today):
    print("\n<<< QUẢN LÝ VÉ ĐÃ BÁN >>>")

    print("\nThao tác: Thêm giao dịch bán vé")
    add_ticket_sale(events, ticket_sales_history, sold_events_today, "EV001", 2)
    add_ticket_sale(events, ticket_sales_history, sold_events_today, "EV002", 5)
    add_ticket_sale(events, ticket_sales_history, sold_events_today, "EV005", 10)
    add_ticket_sale(events, ticket_sales_history, sold_events_today, "EV001", 250)
    add_ticket_sale(events, ticket_sales_history, sold_events_today, "EV999", 1)
    add_ticket_sale(events, ticket_sales_history, sold_events_today, "EV002", -3)

    print("\nThao tác: Kiểm tra sự kiện có vé bán trong ngày")
    check_event_sold_today(sold_events_today, "EV001")
    check_event_sold_today(sold_events_today, "EV003")

    print("\nThao tác: In danh sách giao dịch bán vé (tất cả)")
    print_sales_by_event(ticket_sales_history)

    print("\nThao tác: In danh sách giao dịch bán vé (lọc theo EV001)")
    print_sales_by_event(ticket_sales_history, "EV001")
    print_sales_by_event(ticket_sales_history, "EV004")

    print("\nThao tác: Xóa giao dịch với số lượng vé bằng 0")
    remove_zero_quantity_sales(ticket_sales_history)
    print_sales_by_event(ticket_sales_history)