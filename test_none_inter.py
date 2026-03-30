import mujoco
import mujoco.viewer
import numpy as np
import time
from ik import solve_scara_ik
from interpolation import interpolate_trapezoidal


# 1. Load model (Đảm bảo file .xml có khớp type="slide")
model = mujoco.MjModel.from_xml_path('scene.xml')
data = mujoco.MjData(model)
weld_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, "vaccum_plate")

box_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "bos_pos")
end_effector = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "touch_sensor_point")
# vacuum_plate_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, "vacuum_gripper")
box_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "object_to_pick")

def get_joint_id(name):
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
j3_idx = model.jnt_qposadr[get_joint_id("joint3")] 

with mujoco.viewer.launch_passive(model, data) as viewer:
    
    
    state = -1
    waiting_duration = 0.25 
    vacuum_on = False
    release_time = 0

    #Biến cho motion profile
    start_q = np.zeros(3)
    target_q = np.zeros(3)
    motion_duration = 1
    motion_start_time = 0

    
    while viewer.is_running():
        print(state, data.time)
        step_start = time.time()

        obj_pos = data.site_xpos[box_site_id]

        if state == -1:

            if data.time > waiting_duration:
                state = 0
                
        
        elif state == 0:
            q1, q2, q3 = solve_scara_ik(obj_pos[0], obj_pos[1], 0.1) #Giữ độ cao 0.1
            data.ctrl[0], data.ctrl[1], data.ctrl[2] = q1, q2, 0.06
            print(q1, q2)
            #Tính k/c Euclid 
            dist = np.linalg.norm(data.site_xpos[end_effector][:2] - obj_pos[:2])
            if dist < 0.01: state = 1
        
        elif state == 1:
            data.ctrl[2] = -0.2
            dist_z = abs(data.site_xpos[end_effector][2] - obj_pos[2])
            if dist_z < 0.022: 
                state = 2
            print("stat 1 dist: ", dist_z)
  
        elif state == 2:
            vacuum_on = True
            print("Object is helded")
            state = 3

        elif state == 3:
            data.ctrl[2] = 0.06
            ##idx cua j3 trong qpos
            if data.qpos[j3_idx] > 0.05:
                state = 4
            
        elif state == 4: 
            q1, q2, q3 = solve_scara_ik(0.68, 0, 0.1) #về Goal
            data.ctrl[0], data.ctrl[1], data.ctrl[2] = q1, q2, 0.06

            goal = np.array([0.68, 0])
            ee_xy = data.site_xpos[end_effector][:2]
            if np.linalg.norm(ee_xy - goal) < 0.01:
                release_time = data.time
                state = 5
        elif state == 5:
            data.ctrl[2] = -0.06
            if data.time - release_time > 0.2:
                print("Release state")
                vacuum_on = False


        if vacuum_on:
            data.ctrl[3] = 20
        else:
            data.ctrl[3] = 0


        mujoco.mj_step(model, data)
        viewer.sync()

        # Duy trì tốc độ thời gian thực
        time_until_next_step = model.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)