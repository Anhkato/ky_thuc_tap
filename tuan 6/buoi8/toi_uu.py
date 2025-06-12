import numpy as np
from scipy import optimize

def toi_uu_gio(kq_hoiquy, so_tv=5, tong_gio=200, gio_min=30):
    """Đề xuất phân bổ giờ làm tối ưu dựa trên kết quả hồi quy."""
    print("\n--- TỐI ƯU HÓA GIỜ LÀM ---")
    
    def ham_muc_tieu(h):
        return -np.sum(kq_hoiquy.slope * h + kq_hoiquy.intercept)

    rang_buoc = [{'type': 'ineq', 'fun': lambda h: tong_gio - np.sum(h)}]
    
    gioi_han = optimize.Bounds(lb=gio_min, ub=tong_gio)
    
    ket_qua = optimize.minimize(
        ham_muc_tieu, 
        np.full(so_tv, tong_gio / so_tv), 
        method='SLSQP', 
        bounds=gioi_han, 
        constraints=rang_buoc
    )

    if ket_qua.success:
        print("\n Đề xuất phân bổ giờ làm cho tuần tới:")
        for i, h in enumerate(ket_qua.x):
            print(f"- Thành viên {i+1}: {h:.2f} giờ")
    else:
        print("\n Không tìm thấy giải pháp tối ưu.")