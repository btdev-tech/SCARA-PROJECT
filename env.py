import time
import numpy as np
import cv2
import mujoco
import mujoco.viewer
from color_dectection import dectect_color
import pyautogui

class SCARA_ENV():
    def __init__(self, xml_path):
        self.viewer = None
        self.image = None
        self.current_object_pos = []
        self.current_object_vel = []
   
        self.xml_path = xml_path
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        self.box_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "bos_pos")
        self.end_effector = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "touch_sensor_point")
        self.box_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "object_to_pick")
        self.geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "box") 
        self.j3_idx = self.model.jnt_qposadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "joint3")] 

        self.renderer = mujoco.Renderer(self.model, height=480, width=480)

    def get_ID_List(self):
        return {"box_site_id": self.box_site_id,
            "end_effector": self.end_effector,
            "geom_id": self.geom_id,
            "j3_idx": self.j3_idx
            }
        
    def get_image(self):
        mujoco.mj_forward(self.model, self.data)
        self.renderer.update_scene(self.data, camera="top_view")
        image = self.renderer.render()
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image = cv2.flip(image, 0)
        return image
    
    def detect_color(self):
        raw_frame = self.get_image()
        output = dectect_color(self.get_image())
        if output is not None:
            results, debug_frame = output
            cv2.imshow("Camera View", debug_frame)
            return results
        else:
            # Nếu lỡ bị None, hiển thị khung hình trống hoặc cũ để không crash
            cv2.imshow("Camera View", raw_frame)
            return []

    def step(self, action): ## lưu lịch sử pos/vel
        self.data.ctrl[:] = action
        mujoco.mj_step(self.model, self.data, self.simulation_steps) # 2ms each 
        self.current_object_pos = self.data.qpos[self.object_qpos_addr : self.object_qpos_addr + 3].copy() # x,y,z
        self.current_object_vel = self.data.qvel[self.object_qpos_addr : self.object_qpos_addr + 3].copy() # dx,dy,dz


    def render(self):
        if self.viewer is None:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            pyautogui.press('tab')
            pyautogui.press('h')
        if self.viewer.is_running():
            mujoco.mj_step(self.model, self.data) # 2ms each 
            self.viewer.opt.frame = mujoco.mjtFrame.mjFRAME_NONE
            self.viewer.opt.label = mujoco.mjtLabel.mjLABEL_NONE
            self.viewer.sync()