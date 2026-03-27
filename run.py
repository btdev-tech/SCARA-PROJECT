import mujoco
import mujoco.viewer
import numpy as np

model = mujoco.MjModel.from_xml_path('test robot.xml')
data = mujoco.MjData(model)

# Lấy ID của constraint và các đối tượng cần thiết
constraint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, 'vacuum_contact')
site_id = model.site('suction_site').id
box_id = model.body('box').id

def toggle_vacuum(active):
    """Bật hoặc tắt lực hút chân không"""
    if active:
        # 1. Bật trạng thái active của weld constraint
        model.eq_active[constraint_id] = 1
        
        # 2. Cực kỳ quan trọng: Cập nhật vị trí tương đối (relpose) 
        # Để vật thể dính ngay tại vị trí hiện tại của đầu hút
        # Nếu không có dòng này, vật sẽ bị "giật" về vị trí mặc định lúc khởi tạo XML
        mujoco.mj_set_item_eq_pweld(model, constraint_id, data.site_xpos[site_id], data.site_xquat[site_id], box_id)
    else:
        # Tắt lực hút
        model.eq_active[constraint_id] = 0

# --- Chạy mô phỏng ---
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        # Ví dụ logic: Nếu khoảng cách < 0.05 thì hút
        dist = np.linalg.norm(data.site_xpos[site_id] - data.body('box').xpos)
        
        if dist < 0.05:
            toggle_vacuum(True)
        
        # Nhấn phím hoặc điều kiện khác để thả (ví dụ sau 5 giây)
        # toggle_vacuum(False)

        mujoco.mj_step(model, data)
        viewer.sync()