import numpy as np

# Thông số đã tính từ XML của bạn
L1 = 0.36
L2 = 0.32

def solve_scara_ik(x, y, z_target):
    # 1. Tính khớp 2 (q2) - Elbow
    cos_q2 = (x**2 + y**2 - L1**2 - L2**2) / (2 * L1 * L2)
    cos_q2 = np.clip(cos_q2, -1.0, 1.0)
    q2 = np.arccos(cos_q2)
    
    # 2. Tính khớp 1 (q1) - Shoulder
    q1 = np.arctan2(y, x) - np.arctan2(L2 * np.sin(q2), L1 + L2 * np.cos(q2))
    
    # 3. Tính khớp 3 (z) - Trục Slide
    # Lưu ý: Robot của bạn cao khoảng 0.48m, vật ở 0.3m. 
    # q3 trong MuJoCo là độ dời so với vị trí mặc định của khớp slide.
    q3 = z_target - 0.40  # Điều chỉnh offset tùy theo vị trí home của khớp
    
    return q1, q2, q3