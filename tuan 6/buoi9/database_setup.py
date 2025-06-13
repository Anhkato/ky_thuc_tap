# database_setup.py
import mysql.connector

def thiet_lap_database(cursor, db_name):
  
    try:
        cursor.execute(f"CREATE DATABASE IF NOT EXISTS {db_name} DEFAULT CHARACTER SET 'utf8'")
        print(f"Cơ sở dữ liệu '{db_name}' đã được tạo hoặc đã tồn tại.")
        cursor.execute(f"USE {db_name}")

        create_members_table_query = """
        CREATE TABLE IF NOT EXISTS members (
            member_id INT AUTO_INCREMENT PRIMARY KEY,
            name VARCHAR(100) NOT NULL,
            role VARCHAR(50)
        ) ENGINE=InnoDB;
        """
        cursor.execute(create_members_table_query)
        print("Bảng 'members' đã được tạo hoặc đã tồn tại.")

        create_progress_table_query = """
        CREATE TABLE IF NOT EXISTS weekly_progress (
            progress_id INT AUTO_INCREMENT PRIMARY KEY,
            member_id INT NOT NULL,
            week_number INT NOT NULL,
            hours_worked FLOAT CHECK (hours_worked >= 0),
            tasks_completed INT,
            notes TEXT,
            FOREIGN KEY (member_id) REFERENCES members(member_id) ON DELETE CASCADE
        ) ENGINE=InnoDB;
        """
        cursor.execute(create_progress_table_query)
        print("Bảng 'weekly_progress' đã được tạo hoặc đã tồn tại.")
    except mysql.connector.Error as err:
        print(f"Lỗi khi thiết lập database: {err}")
        exit(1)

def don_dep_database(cursor):
  
    # Xóa bảng weekly_progress
    
    try:
        print("\n--- Bắt đầu dọn dẹp ---")
        cursor.execute("DROP TABLE IF EXISTS weekly_progress")
        print("Bảng 'weekly_progress' đã được xóa.")
        cursor.execute("DROP TABLE IF EXISTS members")
        print("Bảng 'members' đã được xóa.")
    except mysql.connector.Error as err:
        print(f"Lỗi trong quá trình dọn dẹp database: {err}")
