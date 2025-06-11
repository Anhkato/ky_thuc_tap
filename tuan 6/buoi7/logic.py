# logic_phan_tich.py
import pandas as pd
import matplotlib.pyplot as plt
import os

def tao_du_lieu():
   
    data = {
        'Tên': ['An', 'Bình', 'Chi', 'Dũng', 'Giang',
                'An', 'Bình', 'Chi', 'Dũng', 'Giang',
                'An', 'Bình', 'Chi', 'Dũng', 'Giang'],
        'Tuần': [1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3],
        'Bài tập': [5, 4, 6, 3, 5, 7, 5, 8, 5, 6, 8, 7, 7, 6, 8],
        'Điểm': [8.5, 7.0, 9.0, 6.5, 8.0, 9.0, 8.0, 9.5, 7.5, 8.5, 9.5, 8.5, 8.0, 8.0, 9.0]
    }
    df = pd.DataFrame(data)
    df.to_csv('progress.csv', index=False)
    print(" Đã tạo và lưu dữ liệu vào 'progress.csv'.")

def phan_tich_hang_tuan():

    try:
        df = pd.read_csv('progress.csv')
    except FileNotFoundError:
        print("Lỗi: Không tìm thấy tệp 'progress.csv'.")
        return

    print("\n--- PHÂN TÍCH TIẾN ĐỘ HÀNG TUẦN ---")
    cac_nhom_tuan = df.groupby('Tuần')
    for tuan, nhom in cac_nhom_tuan:
        bt_trung_binh = nhom['Bài tập'].mean()
        diem_trung_binh = nhom['Điểm'].mean()
        hoc_vien_top_idx = nhom['Điểm'].idxmax()
        thong_tin_top = df.loc[hoc_vien_top_idx]
        
        print(f"\nPhân tích tuần {tuan}:")
        print(f"  - Bài tập trung bình: {bt_trung_binh:.2f}")
        print(f"  - Điểm trung bình: {diem_trung_binh:.2f}")
        print(f"  - Học viên xuất sắc: {thong_tin_top['Tên']} ({thong_tin_top['Điểm']} điểm)")
        
    print("\n--- HỌC VIÊN HOÀN THÀNH TRÊN 4 BÀI TẬP ---")
    hoc_vien_cham_chi = df[df['Bài tập'] > 4]
    print(hoc_vien_cham_chi)

def truc_quan_hoa_tien_do():
   
    try:
        df = pd.read_csv('progress.csv')
    except FileNotFoundError:
        print(" Lỗi: Không tìm thấy tệp 'progress.csv'.")
        return
    #biểu đồ đường
    plt.figure(figsize=(10, 6))
    for ten, nhom in df.groupby('Tên'):
        plt.plot(nhom['Tuần'], nhom['Điểm'], marker='o', label=ten)
    plt.title('Xu Hướng Điểm Trung Bình Qua Các Tuần')
    plt.xlabel('Tuần')
    plt.ylabel('Điểm Trung Bình')
    plt.xticks(df['Tuần'].unique())
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.savefig('reports/bieu_do_xu_huong.png')
    plt.close()
    print("Đã lưu biểu đồ xu hướng điểm tại 'reports/bieu_do_xu_huong.png'.")
    
    # Biểu đồ cột
    plt.figure(figsize=(8, 6))
    bt_tb_moi_tuan = df.groupby('Tuần')['Bài tập'].mean()
    bt_tb_moi_tuan.plot(kind='bar', color=['skyblue', 'lightgreen', 'salmon'])
    plt.title('Số Bài Tập Hoàn Thành Trung Bình Mỗi Tuần')
    plt.xlabel('Tuần')
    plt.ylabel('Số Bài Tập Trung Bình')
    plt.xticks(rotation=0)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig('reports/bieu_do_so_sanh.png')
    plt.close()
    print("Đã lưu biểu đồ so sánh bài tập tại 'reports/bieu_do_so_sanh.png'.")

def tao_bao_cao_tong_ket():
   
    try:
        df = pd.read_csv('progress.csv')
    except FileNotFoundError:
        print(" Lỗi: Không tìm thấy tệp 'progress.csv'.")
        return

    tong_hop = df.groupby('Tên').agg(Tong_Bai_Tap=('Bài tập', 'sum'), Diem_Trung_Binh=('Điểm', 'mean')).reset_index()
    diem_tuan_1 = df[df['Tuần'] == 1].set_index('Tên')['Điểm']
    tuan_cuoi = df['Tuần'].max()
    diem_tuan_cuoi = df[df['Tuần'] == tuan_cuoi].set_index('Tên')['Điểm']
    su_tien_bo = diem_tuan_cuoi - diem_tuan_1
    hv_tien_bo_nhat = su_tien_bo.idxmax()
    diem_tang = su_tien_bo.max()

    # Viết báo cáo văn bản
    with open('reports/bao_cao.txt', 'w', encoding='utf-8') as f:
        f.write("--- BÁO CÁO TỔNG KẾT TIẾN ĐỘ HỌC TẬP ---\n\n")
        f.write("1. Thống kê của từng học viên:\n")
        for idx, dong in tong_hop.iterrows():
            f.write(f"  - {dong['Tên']}:\n    + Tổng bài tập: {dong['Tong_Bai_Tap']}\n    + Điểm TB: {dong['Diem_Trung_Binh']:.2f}\n")
        f.write("\n2. Ghi nhận đặc biệt:\n")
        f.write(f"  - Học viên tiến bộ nhất: {hv_tien_bo_nhat} (tăng {diem_tang:.2f} điểm)\n")
    print(" Đã lưu báo cáo tổng kết tại 'reports/bao_cao.txt'.")
    
    # biểu đồ tròn
    plt.figure(figsize=(8, 8))
    tong_bt_hoc_vien = tong_hop.set_index('Tên')['Tong_Bai_Tap']
    plt.pie(tong_bt_hoc_vien, labels=tong_bt_hoc_vien.index, autopct='%1.1f%%', startangle=90)
    plt.title('Tỷ Lệ Đóng Góp Bài Tập Của Từng Học Viên')
    plt.ylabel('')
    plt.savefig('reports/bieu_do_dong_gop.png')
    plt.close()
    print(" Đã lưu biểu đồ tỷ lệ đóng góp tại 'reports/bieu_do_dong_gop.png'.")