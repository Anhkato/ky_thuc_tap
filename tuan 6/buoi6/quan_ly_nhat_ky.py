import os

THU_MUC_NHAT_KY = "nhat_ky"
import os

THU_MUC_LOG = "nhat_ky"

def _lay_tep(so_tuan):
    return os.path.join(THU_MUC_LOG, f"tuan_{so_tuan}.txt")

def khoi_tao_thu_muc():
    os.makedirs(THU_MUC_LOG, exist_ok=True)

def luu_nhat_ky(so_tuan, so_gio, so_nhiem_vu, ghi_chu):
    khoi_tao_thu_muc()
    duong_dan = _lay_tep(so_tuan)
    try:
        with open(duong_dan, 'w', encoding='utf-8') as f:
            f.write(f"Tuần: {so_tuan}\n")
            f.write(f"Số giờ làm việc: {so_gio}\n")
            f.write(f"Nhiệm vụ hoàn thành: {so_nhiem_vu}\n")
            f.write(f"Ghi chú: {ghi_chu}\n")
        return True
    except IOError:
        return False

def doc_nhat_ky(so_tuan):
    duong_dan = _lay_tep(so_tuan)
    if os.path.exists(duong_dan):
        with open(duong_dan, 'r', encoding='utf-8') as f:
            return f.read()
    return None

def xoa_nhat_ky(so_tuan):
    duong_dan = _lay_tep(so_tuan)
    try:
        os.remove(duong_dan)
        return True
    except FileNotFoundError:
        return False

def tao_bao_cao():
    khoi_tao_thu_muc()
    tong_tuan = 0
    tong_gio = 0.0
    tong_nhiem_vu = 0

    ten_tep_logs = [f for f in os.listdir(THU_MUC_LOG) if f.startswith('tuan_') and f.endswith('.txt')]

    for ten_tep in ten_tep_logs:
        duong_dan = os.path.join(THU_MUC_LOG, ten_tep)
        try:
            with open(duong_dan, 'r', encoding='utf-8') as f:
                dong = f.readlines()
                tong_gio += float(dong[1].split(':')[1].strip())
                tong_nhiem_vu += int(dong[2].split(':')[1].strip())
                tong_tuan += 1
        except (IOError, IndexError, ValueError):
            continue 
            
    return {"tuan": tong_tuan, "gio": tong_gio, "nhiem_vu": tong_nhiem_vu}

def _lay_duong_dan_tep(so_tuan):
    return os.path.join(THU_MUC_NHAT_KY, f"tuan_{so_tuan}.txt")

def dam_bao_thu_muc_ton_tai():
    os.makedirs(THU_MUC_NHAT_KY, exist_ok=True)

def tao_nhat_ky_tuan(so_tuan, so_gio_lam, so_nhiem_vu, ghi_chu):
    dam_bao_thu_muc_ton_tai()
    duong_dan_tep = _lay_duong_dan_tep(so_tuan)
    try:
        with open(duong_dan_tep, 'w', encoding='utf-8') as f:
            f.write(f"Tuần: {so_tuan}\n")
            f.write(f"Số giờ làm việc: {so_gio_lam}\n")
            f.write(f"Nhiệm vụ hoàn thành: {so_nhiem_vu}\n")
            f.write(f"Ghi chú: {ghi_chu}\n")
        return True
    except IOError as e:
        print(f"Lỗi khi ghi tệp: {e}")
        return False

def doc_nhat_ky_tuan(so_tuan):
    duong_dan_tep = _lay_duong_dan_tep(so_tuan)
    if os.path.exists(duong_dan_tep):
        with open(duong_dan_tep, 'r', encoding='utf-8') as f:
            return f.read()
    else:
        return None

def xoa_nhat_ky_tuan(so_tuan):
    duong_dan_tep = _lay_duong_dan_tep(so_tuan)
    try:
        os.remove(duong_dan_tep)
        return True
    except FileNotFoundError:
        return False

def tao_bao_cao_tong_ket():
    dam_bao_thu_muc_ton_tai()
    tong_so_tuan = 0
    tong_so_gio = 0.0
    tong_so_nhiem_vu = 0

    danh_sach_tep = [f for f in os.listdir(THU_MUC_NHAT_KY) if f.startswith('tuan_') and f.endswith('.txt')]

    for ten_tep in danh_sach_tep:
        duong_dan_tep = os.path.join(THU_MUC_NHAT_KY, ten_tep)
        try:
            with open(duong_dan_tep, 'r', encoding='utf-8') as f:
                cac_dong = f.readlines()
                
                so_gio = float(cac_dong[1].split(':')[1].strip())
                so_nhiem_vu = int(cac_dong[2].split(':')[1].strip())
                
                tong_so_gio += so_gio
                tong_so_nhiem_vu += so_nhiem_vu
                tong_so_tuan += 1
        except (IOError, IndexError, ValueError) as e:
            print(f"Bỏ qua tệp bị lỗi {ten_tep}: {e}")
            
    return {
        "tuan": tong_so_tuan,
        "gio": tong_so_gio,
        "nhiem_vu": tong_so_nhiem_vu
    }