import mysql.connector

def tao_bao_cao_tong_ket(cursor):
   
    print("\n--- Báo cáo tổng kết toàn bộ dự án ---")
    summary_query = """
    SELECT m.name, SUM(p.hours_worked) as total_hours, SUM(p.tasks_completed) as total_tasks
    FROM weekly_progress p
    JOIN members m ON p.member_id = m.member_id
    GROUP BY m.member_id, m.name
    ORDER BY total_hours DESC;
    """
    try:
        cursor.execute(summary_query)
        results = cursor.fetchall()
        if not results:
            print("Chưa có dữ liệu để tạo báo cáo.")
            return

        for name, total_hours, total_tasks in results:
            print(f"\t- {name}: Tổng {total_hours or 0} giờ, {total_tasks or 0} nhiệm vụ")
    except mysql.connector.Error as err:
        print(f"Lỗi khi tạo báo cáo tổng kết: {err}")
