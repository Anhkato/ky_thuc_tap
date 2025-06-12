import numpy as np

def tao_dulieu(ten_tep='hieusuat.npy'):
    """Tạo và lưu dữ liệu hiệu suất giả lập."""
    print(f"Đang tạo dữ liệu và lưu vào tệp '{ten_tep}'...")
    
    so_tuan = 4
    so_thanh_vien = 5
    
    gio_lam = 40 + np.random.randn(so_tuan, so_thanh_vien, 1) * 3
    nhiem_vu = np.round(gio_lam / 8 + np.random.randn(so_tuan, so_thanh_vien, 1) * 0.5)
    
    du_lieu_hieu_suat = np.concatenate((gio_lam, nhiem_vu), axis=2)
    
    du_lieu_hieu_suat[2, 3, 0] = 60.5
    du_lieu_hieu_suat[du_lieu_hieu_suat < 0] = 0
    
    np.save(ten_tep, du_lieu_hieu_suat)
    print("-> Đã tạo dữ liệu thành công.")