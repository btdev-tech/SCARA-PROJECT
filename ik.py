import numpy as np

# Length of first and second joint
L1 = 0.36
L2 = 0.32

def solve_scara_ik(x, y, z_target):
    # 1. joint 2 (q2) - Elbow
    cos_q2 = (x**2 + y**2 - L1**2 - L2**2) / (2 * L1 * L2)
    cos_q2 = np.clip(cos_q2, -1.0, 1.0)
    
    q2 = np.arccos(cos_q2)
    if y < 0: q2 = -q2
    
    # 2. joint 1 (q1) - Shoulder
    q1 = np.arctan2(y, x) - np.arctan2(L2 * np.sin(q2), L1 + L2 * np.cos(q2))
    
    # 3. joint 3 (z) - Slider
    q3 = z_target - 0.40 
    
    return q1, q2, q3