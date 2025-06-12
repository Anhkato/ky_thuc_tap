import numpy as np

from data import tao_dulieu
from phantich import pt_coban, pt_nangcao
from toi_uu import toi_uu_gio

def chinh():
    """Hàm chính điều phối toàn bộ chương trình."""
    TEN_TEP = 'hieusuat.npy'
    
    tao_dulieu(TEN_TEP)

    try:
        data = np.load(TEN_TEP)
        print(f"-> Đã đọc dữ liệu từ '{TEN_TEP}'.")
        
        pt_coban(data)
        kq_hoiquy = pt_nangcao(data)
        
        if kq_hoiquy and kq_hoiquy.slope > 0:
            toi_uu_gio(kq_hoiquy, so_tv=data.shape[1])

    except FileNotFoundError:
        print(f"\nLỗi: Không tìm thấy tệp '{TEN_TEP}'.")
    except Exception as e:
        print(f"\nĐã xảy ra lỗi: {e}")

if __name__ == "__main__":
    chinh()