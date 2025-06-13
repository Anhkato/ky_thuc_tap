import mysql.connector

def them_du_lieu(cursor, cnx):
   
    print("\n--- Bắt đầu thêm dữ liệu mẫu ---")
    try:
        cursor.execute("DELETE FROM weekly_progress")
        cursor.execute("DELETE FROM members")
        cursor.execute("ALTER TABLE members AUTO_INCREMENT = 1")
        
        members_data = [
            ("Nguyễn Văn An", "Developer"), ("Trần Thị Bình", "Designer"),
            ("Lê Minh Cường", "Project Manager"), ("Phạm Thị Dung", "Tester"),
            ("Hoàng Văn Em", "Developer")
        ]
        add_member_query = "INSERT INTO members (name, role) VALUES (%s, %s)"
        cursor.executemany(add_member_query, members_data)
        
        progress_data = [
            (1, 1, 40.5, 5, "Hoàn thành module A"), (2, 1, 35.0, 3, "Thiết kế giao diện trang chủ"),
            (3, 1, 45.0, 2, "Họp và lên kế hoạch"), (4, 1, 38.0, 10, "Kiểm thử module B"),
            (5, 1, 42.0, 6, "Phát triển tính năng C"), (1, 2, 38.0, 4, "Sửa lỗi module A"),
            (2, 2, 40.0, 4, "Hoàn thiện wireframe"), (3, 2, 42.5, 3, "Báo cáo và đánh giá"),
            (4, 2, 39.5, 12, "Viết test case cho module C"), (5, 2, 41.0, 5, "Tối ưu hóa hiệu năng")
        ]
        add_progress_query = "INSERT INTO weekly_progress (member_id, week_number, hours_worked, tasks_completed, notes) VALUES (%s, %s, %s, %s, %s)"
        cursor.executemany(add_progress_query, progress_data)
        
        cnx.commit()
        print(f"Đã thêm {cursor.rowcount} bản ghi tiến độ.")
    except mysql.connector.Error as err:
        print(f"Lỗi khi thêm dữ liệu: {err}")
        cnx.rollback()

def truy_van_tien_do(cursor, week_num):
   
    print(f"\n--- Báo cáo tiến độ Tuần {week_num} ---")
    query = """
    SELECT m.name, p.hours_worked, p.tasks_completed, p.notes
    FROM weekly_progress p
    JOIN members m ON p.member_id = m.member_id
    WHERE p.week_number = %s
    ORDER BY p.tasks_completed DESC
    LIMIT 5;
    """
    try:
        cursor.execute(query, (week_num,))
        results = cursor.fetchall()
        if not results:
            print(f"Không tìm thấy dữ liệu cho tuần {week_num}.")
            return
            
        for name, hours, tasks, notes in results:
            print(f"\t- {name}: {hours} giờ, {tasks} nhiệm vụ, Ghi chú: {notes}")
    except mysql.connector.Error as err:
        print(f"Lỗi khi truy vấn dữ liệu: {err}")

def cap_nhat_tien_do(cursor, cnx, progress_id, new_hours, new_notes):
 
    print(f"\n--- Cập nhật tiến độ cho bản ghi ID: {progress_id} ---")
    update_query = """
    UPDATE weekly_progress
    SET hours_worked = %s, notes = %s
    WHERE progress_id = %s;
    """
    try:
        cursor.execute(update_query, (new_hours, new_notes, progress_id))
        cnx.commit()
        if cursor.rowcount > 0:
            print(f"Đã cập nhật thành công bản ghi có progress_id = {progress_id}.")
        else:
            print(f"Không tìm thấy bản ghi nào với progress_id = {progress_id}.")
    except mysql.connector.Error as err:
        print(f"Lỗi khi cập nhật dữ liệu: {err}")
        cnx.rollback()

def xoa_tien_do(cursor, cnx, week_num):
 
    print(f"\n--- Xóa dữ liệu tiến độ Tuần {week_num} ---")
    delete_query = "DELETE FROM weekly_progress WHERE week_number = %s"
    try:
        cursor.execute(delete_query, (week_num,))
        affected_rows = cursor.rowcount
        cnx.commit()
        if affected_rows > 0:
            print(f"Đã xóa thành công {affected_rows} bản ghi của tuần {week_num}.")
        else:
            print(f"Không có bản ghi nào của tuần {week_num} để xóa.")
    except mysql.connector.Error as err:
        print(f"Lỗi khi xóa dữ liệu: {err}")
        cnx.rollback()
