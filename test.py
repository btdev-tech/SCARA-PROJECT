import mujoco
import mujoco.viewer
import numpy as np
import time

# 1. Load model (Đảm bảo file .xml có khớp type="slide")
model = mujoco.MjModel.from_xml_path('scene.xml')
data = mujoco.MjData(model)

with mujoco.viewer.launch_passive(model, data) as viewer:
    print("Đang test Khớp Trượt (Slider)...")
    
    while viewer.is_running():
        step_start = time.time()

        # Tạo chuyển động lên xuống
        # np.sin tạo giá trị từ -1 đến 1
        # Ta nhân với 0.05 để trục trượt trong khoảng +- 5cm
        # Trừ đi 0.1 để nó hoạt động chủ yếu ở phía dưới cánh tay
       
        # position = np.array([3.14,1.0, -0.06])
        # Gửi lệnh vào Actuator của khớp trượt (thường là index 2)
        # Kiểm tra thứ tự actuator trong file XML của bạn nhé!
        # data.ctrl[:]= position
        

        mujoco.mj_step(model, data)
        viewer.sync()

        # Duy trì tốc độ thời gian thực
        time_until_next_step = model.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)