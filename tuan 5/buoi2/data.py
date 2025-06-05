def initialize_events():
    events = [
        {"id": "EV001", "name": "Hội chợ sách mùa hè", "ticket_price": 50000.0, "tickets_left": 200},
        {"id": "EV002", "name": "Triển lãm nghệ thuật đương đại", "ticket_price": 75000.0, "tickets_left": 150},
        {"id": "EV003", "name": "Workshop làm gốm", "ticket_price": 120000.0, "tickets_left": 50},
        {"id": "EV004", "name": "Đêm nhạc acoustic", "ticket_price": 100000.0, "tickets_left": 300},
        {"id": "EV005", "name": "Phiên chợ đồ handmade", "ticket_price": 30000.0, "tickets_left": 250}
    ]
    return events

def initialize_sponsors():
    sponsors = {
        "SP001": ("Công ty TNHH Sách Việt", 50000000.0),
        "SP002": ("Quỹ Văn Hóa ABC", 120000000.0),
        "SP003": ("Tập đoàn Sáng Tạo XYZ", 75000000.0)
    }
    return sponsors

def initialize_ticket_sales_history():
    return []

def initialize_sold_events_today():
    return set()