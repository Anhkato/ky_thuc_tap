import numpy as np

def find_low_stock_events(events, threshold=20):
    low_stock = [event for event in events if event["tickets_left"] < threshold]
    print("\n--- Sự kiện sắp hết vé (dưới " + str(threshold) + " vé) ---")
    if not low_stock:
        print("Không có sự kiện nào sắp hết vé.")
        return []
    for event in low_stock:
        print(f"  Tên sự kiện: {event['name']}, Vé còn lại: {event['tickets_left']}")
    return low_stock

def calculate_total_remaining_ticket_value(events):
    if not events:
        return 0.0
    event_prices = np.array([event["ticket_price"] for event in events])
    event_tickets_left = np.array([event["tickets_left"] for event in events])
    total_value = np.sum(event_prices * event_tickets_left)
    print(f"\nTổng giá trị vé còn lại từ tất cả sự kiện: {total_value:,.0f} VNĐ")
    return total_value

def list_unique_sold_event_ids(ticket_sales_history):
    if not ticket_sales_history:
        print("\nChưa có lịch sử bán vé.")
        return set()
    unique_event_ids = set(sale["event_id"] for sale in ticket_sales_history)
    print("\n--- Các mã sự kiện duy nhất đã bán vé ---")
    if not unique_event_ids:
        print("Chưa có sự kiện nào được bán vé.")
    else:
        for event_id in unique_event_ids:
            print(f"  {event_id}")
    return unique_event_ids

def generate_report(events, ticket_sales_history):
    print("\n\n========== BÁO CÁO THỐNG KÊ VĂN HÓA CỘNG ĐỒNG ==========")
    low_stock_events = find_low_stock_events(events, 50)
    total_remaining_value = calculate_total_remaining_ticket_value(events)
    sold_event_ids = list_unique_sold_event_ids(ticket_sales_history)

    print("\n--- Tóm tắt báo cáo ---")
    if low_stock_events:
        print("Sự kiện sắp hết vé:")
        for event in low_stock_events:
            print(f"  - {event['name']} (còn {event['tickets_left']} vé)")
    else:
        print("Sự kiện sắp hết vé: Không có.")

    print(f"Tổng giá trị vé còn lại: {total_remaining_value:,.0f} VNĐ")

    if sold_event_ids:
        print("Mã sự kiện đã bán vé: " + ", ".join(sorted(list(sold_event_ids))))
    else:
        print("Sự kiện đã bán vé: Chưa có.")
    print("==========================================================")