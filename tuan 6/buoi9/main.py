# main.py
import mysql.connector
from mysql.connector import errorcode

import config
import database_setup
import data_manager
import reporting

def main():
   
    cnx = None  
    try:
        cnx = mysql.connector.connect(**config.DB_CONFIG)
        cursor = cnx.cursor()

        database_setup.thiet_lap_database(cursor, config.DB_NAME)

        data_manager.them_du_lieu(cursor, cnx)

        data_manager.truy_van_tien_do(cursor, 1)

        data_manager.cap_nhat_tien_do(cursor, cnx, 1, 45.0, "Hoàn thành sớm và review code")
        data_manager.truy_van_tien_do(cursor, 1)

        reporting.tao_bao_cao_tong_ket(cursor)

        data_manager.xoa_tien_do(cursor, cnx, 2)
        data_manager.truy_van_tien_do(cursor, 2)
        
        reporting.tao_bao_cao_tong_ket(cursor)


    except mysql.connector.Error as err:
        if err.errno == errorcode.ER_ACCESS_DENIED_ERROR:
            print("Lỗi: Sai tên người dùng hoặc mật khẩu MySQL.")
        elif err.errno == errorcode.ER_BAD_DB_ERROR:
            print(f"Lỗi: Cơ sở dữ liệu '{config.DB_NAME}' không tồn tại.")
        else:
            print(f"Lỗi MySQL: {err}")
    finally:
        if cnx and cnx.is_connected():
            cursor.close()
            cnx.close()
            print("\nĐã đóng kết nối MySQL.")

if __name__ == "__main__":
    main()
