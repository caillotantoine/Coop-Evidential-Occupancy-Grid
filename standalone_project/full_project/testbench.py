from agent import Agent
from vector import vec2, vec3, vec4
from Tmat import TMat
from bbox import Bbox2D, Bbox3D
from tqdm import tqdm
import json
from typing import List


dataset_path:str = '/home/caillot/Documents/Dataset/CARLA_Dataset_B'
agents:List[Agent] = []
with open(f"{dataset_path}//information.json") as json_file:
    info = json.load(json_file)
    agent_l = info['agents']
    agents = [Agent(dataset_path=dataset_path, id=idx) for idx, agent in enumerate(agent_l) if agent['type'] != "pedestrian"]

for idx, agent in enumerate(agents):
    print(f"{idx} : \t{agent}")


a = agents[6]
visible_bbox:Bbox2D = a.get_visible_bbox(frame=56)
print(visible_bbox)



# for agent in agents:
#     print(agent.get_visible_bbox(56))


# v0 = Agent(dataset_path=dataset_path, id=18)
# print(v0)
# # v0.get_state(56)
# for i in tqdm(range(1, 100)):
#     v0.get_visible_bbox(i)