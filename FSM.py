import numpy as np
from ik import solve_scara_ik

class FSM():
    def __init__(self, mjModel, mjData, ID_List):
        self.mjModel = mjModel
        self.data = mjData
        self.ID_List = ID_List
        
        self.state = "RESET"
        self.waiting_duration = 0.5
        self.release_time = 0
        
        self.results = []
        self.target_obj = None
        self.obj_x = 0
        self.obj_y = 0
        self.obj_pos = []
        self.colors = [
            [1, 0, 0, 0.5], # RED
            [0, 1, 0, 0.5], # GREEN
            [0, 0, 1, 0.5]  # BLUE
        ]
        self.vacuum_on = False
    
    
    def get_target_for_scara(self, cX, cY):
        dx_pixel = cX - 239
        dy_pixel = cY - 239
        ratio = 0.1/173
        obj_x_table = -0.55 - (dx_pixel * ratio)
        obj_y_table = 0 - (dy_pixel * ratio)
        return obj_x_table, obj_y_table

    
    def FSM_Run(self):
        print(self.state)
        if self.state == "RESET":
            self.target_obj = None
            self.state = "DETECT_OBJ"
        
        if len(self.results) > 0 and self.target_obj is None and self.data.time > self.waiting_duration + self.release_time:
            self.target_obj = self.results[0]
            self.obj_x, self.obj_y = self.get_target_for_scara(self.target_obj["center"][0], self.target_obj["center"][1])
            self.obj_pos = [self.obj_x, self.obj_y, 0.26]
            self.state = "MOVE_TO_XY"
  
        if self.state == "MOVE_TO_XY":
            q1, q2, q3 = solve_scara_ik(self.obj_x, self.obj_y, 0.1) 
            self.data.ctrl[0], self.data.ctrl[1], self.data.ctrl[2] = q1, q2, 0.06
            print(q1, q2)
             
            dist = np.linalg.norm(self.data.site_xpos[self.ID_List["end_effector"]][:2] - self.obj_pos[:2])

            if dist < 0.01: self.state = "LOWER_Z_AXIS"
            
        
        elif self.state == "LOWER_Z_AXIS":
            self.data.ctrl[2] = -0.2
            dist_z = abs(self.data.site_xpos[self.ID_List["end_effector"]][2] - self.obj_pos[2])
            if dist_z < 0.022: 
                self.state = "VACUUM_ON"
            print("stat 1 dist: ", dist_z)
  
        elif self.state == "VACUUM_ON":
            self.vacuum_on = True
            print("Object is helded")
            self.state = "LIFT_OBJ"

        elif self.state == "LIFT_OBJ":
            self.data.ctrl[2] = 0.06
            if self.data.qpos[self.ID_List["j3_idx"]] > 0.05:
                self.state = "MOVE_TO_GOAL"
            
        elif self.state == "MOVE_TO_GOAL": 
            
            goal_x = self.target_obj["goal_pos"][0]
            goal_y = self.target_obj["goal_pos"][1]


            q1, q2, q3 = solve_scara_ik(goal_x, goal_y, 0.1) #về Goal
            self.data.ctrl[0], self.data.ctrl[1], self.data.ctrl[2] = q1, q2, 0.06

            goal = np.array([goal_x, goal_y])
            ee_xy = self.data.site_xpos[self.ID_List["end_effector"]][:2]
            print(np.linalg.norm(ee_xy - goal))
            if np.linalg.norm(ee_xy - goal) < 0.01:
                self.release_time = self.data.time
                self.state = "LOWER_AT_GOAL"

        elif self.state == "LOWER_AT_GOAL":
            self.data.ctrl[2] = -0.06
            if self.data.time - self.release_time > 0.2:
                print("Release self.state")
                self.vacuum_on = False
                self.state = "WAIT_FOR_STABILITY"

        elif self.state == "WAIT_FOR_STABILITY":
            if self.data.time - self.release_time > 0.5:
                x_pos = np.random.uniform(-0.55-0.13, -0.55+0.1)
                y_pos = np.random.uniform(-0.13, 0.13)
                quat = np.random.uniform(-1, 1, size=4)
                quat /= np.linalg.norm(quat)
                self.data.qpos[3] = x_pos
                self.data.qpos[4] = y_pos
                self.data.qpos[5] = 0.26
                self.data.qpos[6:10] = quat
                
                random_color = self.colors[np.random.randint(0, len(self.colors))]
                self.mjModel.geom_rgba[self.ID_List["geom_id"]] = random_color

                self.state = "RESET"
                self.release_time += 0.5

        if self.vacuum_on:
            self.data.ctrl[3] = 20
        else:
            self.data.ctrl[3] = 0