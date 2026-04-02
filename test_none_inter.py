import mujoco
import mujoco.viewer
import numpy as np
import time
import pyautogui
from ik import solve_scara_ik
from color_dectection import dectect_color
import cv2

model = mujoco.MjModel.from_xml_path('scene.xml')
data = mujoco.MjData(model)

renderer = mujoco.Renderer(model, height=480, width=480)

box_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "bos_pos")
end_effector = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "touch_sensor_point")
box_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "object_to_pick")
Goal_red = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "Goal_point_red")
Goal_green = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "Goal_point_green")
Goal_blue = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "Goal_point_blue")
geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "box")
    
red_goal=data.qpos[10:12]
green_goal=data.qpos[17:19]
blue_goal=data.qpos[24:26]
def get_image():
    mujoco.mj_forward(model, data)
    renderer.update_scene(data, camera="top_view")
    image = renderer.render()
    
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image = cv2.flip(image, 0)
    return image

def get_target_for_scara(cX, cY):

    dx_pixel = cX - 239
    dy_pixel = cY - 239

    table_size = 0.2

    ratio = 0.1/173
    obj_x_table = -0.55 - (dx_pixel * ratio)
    obj_y_table = 0 - (dy_pixel * ratio)
    return obj_x_table, obj_y_table



def get_joint_id(name):
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
j3_idx = model.jnt_qposadr[get_joint_id("joint3")] 

with mujoco.viewer.launch_passive(model, data) as viewer:
    pyautogui.press('tab')
    pyautogui.press('h')
    viewer.opt.frame = mujoco.mjtFrame.mjFRAME_NONE
    viewer.opt.label = mujoco.mjtLabel.mjLABEL_NONE

    
    state = -1
    waiting_duration = 0.1
    vacuum_on = False
    target_obj = None
    release_time = 0
    colors = [
        [1, 0, 0, 0.5], # Đỏ
        [0, 1, 0, 0.5], # Xanh lá
        [0, 0, 1, 0.5]  # Xanh dương
    ]

    
    while viewer.is_running():
        
        raw_frame = get_image()
        output = dectect_color(get_image(), red_goal, green_goal, blue_goal)
        if output is not None:
            results, debug_frame = output
            cv2.imshow("Camera View", debug_frame)
        else:
            # Nếu lỡ bị None, hiển thị khung hình trống hoặc cũ để không crash
            cv2.imshow("Camera View", raw_frame)

        
        print(state, data.time)
        
        step_start = time.time()

        if state == -1:
            target_obj = None
        
        if len(results) > 0 and target_obj is None and data.time > waiting_duration + release_time:
            target_obj = results[0]
            obj_x, obj_y = get_target_for_scara(target_obj["center"][0], target_obj["center"][1])
            obj_pos = [obj_x, obj_y, 0.26]
            state = 0
  
        if state == 0:
            q1, q2, q3 = solve_scara_ik(obj_x, obj_y, 0.1) #Giữ độ cao 0.1
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
            if data.qpos[j3_idx] > 0.05:
                state = 4
            
        elif state == 4: 
            
            goal_x = target_obj["goal_pos"][0]
            goal_y = target_obj["goal_pos"][1]


            q1, q2, q3 = solve_scara_ik(goal_x, goal_y, 0.1) #về Goal
            data.ctrl[0], data.ctrl[1], data.ctrl[2] = q1, q2, 0.06

            goal = np.array([goal_x, goal_y])
            ee_xy = data.site_xpos[end_effector][:2]
            print(np.linalg.norm(ee_xy - goal))
            if np.linalg.norm(ee_xy - goal) < 0.01:
                release_time = data.time
                state = 5

        elif state == 5:
            data.ctrl[2] = -0.06
            if data.time - release_time > 0.2:
                print("Release state")
                vacuum_on = False
                state = 6

        elif state == 6:
            if data.time - release_time > 0.5:
                x_pos = np.random.uniform(-0.55-0.13, -0.55+0.1)
                y_pos = np.random.uniform(-0.13, 0.13)
                quat = np.random.uniform(-1, 1, size=4)
                quat /= np.linalg.norm(quat)
                data.qpos[3] = x_pos
                data.qpos[4] = y_pos
                data.qpos[5] = 0.26
                data.qpos[6:10] = quat
                
                random_color = colors[np.random.randint(0, len(colors))]
                model.geom_rgba[geom_id] = random_color

                state = -1
                release_time += 0.8

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