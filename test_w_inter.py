import mujoco
import mujoco.viewer
import numpy as np
import time
from ik import solve_scara_ik
from interpolation import interpolate_trapezoidal # Giả định hàm này trả về q tại thời điểm t

# 1. Load model
model = mujoco.MjModel.from_xml_path('scene.xml')
data = mujoco.MjData(model)

# Mapping ID
box_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "bos_pos")
end_effector = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "touch_sensor_point")

def get_joint_id(name):
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)

j3_idx = model.jnt_qposadr[get_joint_id("joint3")] 

with mujoco.viewer.launch_passive(model, data) as viewer:
    
    state = -1
    waiting_duration = 0.25 
    vacuum_on = False
    
    # Biến phục vụ Motion Profile
    start_q = np.zeros(3)      # Lưu vị trí khớp lúc bắt đầu chuyển động
    target_q = np.zeros(3)     # Lưu vị trí đích
    motion_duration = 0.5    # Thời gian di chuyển (giây), chỉnh tùy độ mượt
    motion_start_time = 0.0    # Mốc thời gian bắt đầu di chuyển
    
    while viewer.is_running():
        step_start = time.time()
        obj_pos = data.site_xpos[box_site_id]

        # --- FSM với Motion Profile ---
        print(state)
        if state == -1:
            if data.time > waiting_duration:
                state = 0
                # Khởi tạo cho State 0
                motion_start_time = data.time
                start_q = np.array([data.qpos[0], data.qpos[1], data.qpos[j3_idx]])
                q1, q2, _ = solve_scara_ik(obj_pos[0], obj_pos[1], 0.1)
                target_q = np.array([q1, q2, 0.06])

        elif state == 0: # Di chuyển XY đến trên đầu vật
            t_rel = data.time - motion_start_time
            # Nội suy từng khớp
            data.ctrl[0] = interpolate_trapezoidal(start_q[0], target_q[0], t_rel, motion_duration)
            data.ctrl[1] = interpolate_trapezoidal(start_q[1], target_q[1], t_rel, motion_duration)
            data.ctrl[2] = start_q[2] # Giữ nguyên Z

            if t_rel >= motion_duration:
                print("Đã đến vị trí XY")
                state = 1
                motion_start_time = data.time
                start_q = np.array([data.qpos[0], data.qpos[1], data.qpos[j3_idx]])
                target_q[2] = -0.06 # Target mới cho Z

        elif state == 1: # Hạ Z xuống
            t_rel = data.time - motion_start_time
            data.ctrl[2] = interpolate_trapezoidal(start_q[2], target_q[2], t_rel, 0.5) # Hạ Z nhanh hơn (0.5s)
            
            dist_z = abs(data.site_xpos[end_effector][2] - obj_pos[2])
            if dist_z < 0.005 or t_rel >= 0.5: 
                state = 2

        elif state == 2: # Bật hút
            vacuum_on = True
            print("Đã hút vật")
            state = 3
            motion_start_time = data.time
            start_q = np.array([data.qpos[0], data.qpos[1], data.qpos[j3_idx]])
            target_q[2] = 0.06 # Nhấc lên

        elif state == 3: # Nhấc vật lên
            t_rel = data.time - motion_start_time
            data.ctrl[2] = interpolate_trapezoidal(start_q[2], target_q[2], t_rel, 0.5)
            print(data.qpos[j3_idx])
            if data.qpos[j3_idx] > 0.04:
                state = 4
                motion_start_time = data.time
                start_q = np.array([data.qpos[0], data.qpos[1], data.qpos[j3_idx]])
                q1_g, q2_g, _ = solve_scara_ik(0.68, 0, 0.1)
                target_q = np.array([q1_g, q2_g, 0.06])

        elif state == 4: # Di chuyển về Goal
            t_rel = data.time - motion_start_time
            data.ctrl[0] = interpolate_trapezoidal(start_q[0], target_q[0], t_rel, motion_duration)
            data.ctrl[1] = interpolate_trapezoidal(start_q[1], target_q[1], t_rel, motion_duration)
            
            if t_rel >= motion_duration:
                print("Đã đến Goal, chờ ổn định...")
                release_time = data.time
                motion_start_time = data.time
                start_q_z = data.qpos[j3_idx]
                state = 5

        elif state == 5: # Chờ ổn định 1s trước khi thả
            t_rel = data.time - motion_start_time
            data.ctrl[2] = interpolate_trapezoidal(start_q_z, -0.06, t_rel, 0.5)
            if t_rel >= 0.2:
                print("Đã hạ sát mặt bàn, chờ ổn định...")
                release_time = data.time
                state = 6

        elif state == 6:
            if data.time - release_time > 1.0: # Chờ 1s cho hộp hết rung
                print("Bắt đầu thả")
                vacuum_on = False
                state = 6

        # --- Logic Actuator ---
        data.ctrl[3] = 20 if vacuum_on else 0

        mujoco.mj_step(model, data)
        viewer.sync()

        # Duy trì real-time
        time_until_next_step = model.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)