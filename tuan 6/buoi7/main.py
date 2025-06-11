import logic as logic 
import os

def main():
  
    thu_muc_bao_cao = 'reports'
    if not os.path.exists(thu_muc_bao_cao):
        os.makedirs(thu_muc_bao_cao)
        
    print(" Bắt đầu chương trình phân tích tiến độ học tập...")
    
    logic.tao_du_lieu()
    logic.phan_tich_hang_tuan()
    logic.truc_quan_hoa_tien_do()
    logic.tao_bao_cao_tong_ket()
    
    print(f"\n Chương trình đã hoàn thành. Vui lòng kiểm tra thư mục '{thu_muc_bao_cao}' để xem kết quả.")

if __name__ == "__main__":
    main()