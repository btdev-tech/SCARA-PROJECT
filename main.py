import time
from FSM import FSM
from env import SCARA_ENV

xml_path = 'scene.xml'
env = SCARA_ENV(xml_path)
machine_state = FSM(env.model, env.data, env.get_ID_List())

while True:
    step_start = time.time()
    machine_state.results = env.detect_color()
    machine_state.FSM_Run()
    env.render()
