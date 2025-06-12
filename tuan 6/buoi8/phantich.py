import numpy as np
from scipy import stats

def pt_coban(data):
    """Thực hiện phân tích thống kê cơ bản cho từng tuần."""
    print("\n--- PHÂN TÍCH CƠ BẢN ---")
    for i, dl_tuan in enumerate(data):
        gio = dl_tuan[:, 0]
        nv = dl_tuan[:, 1]
        
        print(f"\n Tuần {i + 1}:")
        print(f"- Giờ làm trung bình: {np.mean(gio):.2f}")
        print(f"- Độ lệch chuẩn giờ: {np.std(gio):.2f}")
        print(f"- Tổng nhiệm vụ: {int(np.sum(nv))}")
        print(f"- NV xuất sắc: TV {np.argmax(nv) + 1} ({int(np.max(nv))} NV)")

def pt_nangcao(data):
    """Thực hiện hồi quy, tính tương quan và tìm ngoại lai."""
    print("\n--- PHÂN TÍCH NÂNG CAO ---")
    
    tat_ca_gio = data[:, :, 0].flatten()
    tat_ca_nv = data[:, :, 1].flatten()
    
    ket_qua_hoiquy = stats.linregress(tat_ca_gio, tat_ca_nv)
    
    nguong = 2 * np.std(tat_ca_gio)
    gio_tb = np.mean(tat_ca_gio)
    ngoai_lai = tat_ca_gio[(tat_ca_gio < gio_tb - nguong) | (tat_ca_gio > gio_tb + nguong)]
    
    print("\n Hồi quy (Giờ làm & Nhiệm vụ):")
    print(f"- Độ dốc: {ket_qua_hoiquy.slope:.4f}")
    print(f"- Tương quan: {ket_qua_hoiquy.rvalue:.4f}")
    
    if ngoai_lai.size > 0:
        print(f"- Giá trị ngoại lai (giờ làm): {np.round(ngoai_lai, 2)}")
    
    return ket_qua_hoiquy