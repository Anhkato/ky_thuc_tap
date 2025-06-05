from data import initialize_events, initialize_sponsors, initialize_ticket_sales_history, initialize_sold_events_today
from event import manage_events
from nha_tai_tro import manage_sponsors
from ticket import process_tickets
from reporter import generate_report

def main():
    """Hàm chính điều phối chương trình."""
    print("Chào mừng đến với Chương trình Quản lý Sự kiện Văn hóa Cộng đồng!")

    print("\n--- KHỞI TẠO DỮ LIỆU ---")
    events = initialize_events()
    sponsors = initialize_sponsors()
    ticket_sales_history = initialize_ticket_sales_history()
    sold_events_today = initialize_sold_events_today()
    print("Khởi tạo dữ liệu thành công.")
    print(f"Số lượng sự kiện ban đầu: {len(events)}")
    print(f"Số lượng nhà tài trợ ban đầu: {len(sponsors)}")


    # hàm Quản lý sự kiện
    manage_events(events)


    # 3. Quản lý nhà tài trợ
    manage_sponsors(sponsors)


    # 4. Quản lý vé đã bán
    process_tickets(events, ticket_sales_history, sold_events_today)


    # 5. Tạo và in báo cáo
    generate_report(events, ticket_sales_history)

    print("\nChương trình kết thúc.")

if __name__ == "__main__":
    main()