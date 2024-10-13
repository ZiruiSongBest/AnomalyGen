from gpt_4.query import query
from gpt_4.prompts.utils import build_task_given_text
from gpt_4.prompts.prompt_distractor import generate_distractor
import time, datetime, os, copy
from manipulation.partnet_category import partnet_categories

user_contents = [
"""
I will provide an articulated object, with its articulation tree and semantics. Your goal is to imagine some dangerous or unsanitary household anomalies that a robotic arm can address with the articulated object. You can think of the robotic arm as a Franka Panda robot. The scenario will be built in a simulator for the robot to learn it. 
Please note that the robot arm has limited capabilities; for example, it only has one arm, making tasks like retracting the blade of a folding knife quite challenging. When setting up tasks to mitigate dangers caused by a folding knife, use simpler methods that are easier for the robot arm to handle, for example, placing the knife in a safe drawer rather than retracting the blade.

Only the articulated object sometimes will have functions, e.g., a microwave can be used to heat food, in these cases, feel free to include other objects that are needed for the task. 
Please do not think of tasks that try to assemble or disassemble the object. Do not think of tasks that aim to clean the object or check its functionality. 
The tasks will be categorized into three kinds.You just need to combine the articulated object I provided and choose a task type to write the task.
Household Hazards:This category focuses on tasks that eliminate hazards in the home, for example, store sharp objects safely, move objects near the edge of the table back to the center and pick up anything that might trip people.
Hygiene Management:This category focuses on tasks that ensure a clean and hygienic environment in the home, for example, dealing with spoiled food or cleaning, close the open refrigerator door to prevent food from spoiling and pick up the trash on the floor.
Child Safety Measures:This category focuses on tasks that remove dangerous objects or substances that could harm children, for example, store medications and similar items, keep knives and sharp objects out of children's reach or Close open large furniture(cabinets, dishwashers,etc.).
Please focus on addressing specific problems that arise in these areas.

For each task you imagined, please simplify this task and write in the following format: 
Task name: the name of the task.
Explanation: Explain why is the scenario unsafe/unsanitary/not safe for children.
Description: some basic descriptions of the tasks.
Additional Objects: Additional objects other than the provided articulated object required for completing the task. 
Links: Links of the articulated objects that are required to perform the task. 
- Link 1: reasons why this link is needed for the task
- Link 2: reasons why this link is needed for the task
- …
Joints: Joints of the articulated objects that are required to perform the task. 
- Joint 1: reasons why this joint is needed for the task
- Joint 2: reasons why this joint is needed for the task
- …

Example Input: 
```Box articulation tree
links: 
base
link_0
link_1
link_2

joints: 
joint_name: joint_0 joint_type: revolute parent_link: link_2 child_link: link_0
joint_name: joint_1 joint_type: revolute parent_link: link_2 child_link: link_1
joint_name: joint_2 joint_type: fixed parent_link: base child_link: link_2

```
```Box semantics
link_0 hinge rotation_lid
link_1 hinge rotation_lid
link_2 free box_body

```

Example Output:
Task name: Put a knife into the box
Explanation: A knife on the floor is dangerous because it can cause injury if someone steps on it.(Unsafe)
Description: The robot arm opens the box first, then picks up the knife, places it inside the box, and finally closes the box.
Additional Objects: Knife
Links:
- link_0: Link_0 is the box lid from the semantics. The robot needs to open the lid in order to put the knife inside the box.
- link_1: Link_1 is also a part of the box lid. The robot needs to ensure this part is also opened to avoid any obstruction while placing the knife inside.
Joints:
- joint_0: From the articulation tree, this is the revolute joint that connects link_0. The robot needs to actuate this joint to open and close the lid.
- joint_1: From the articulation tree, joint_1 connects link_1, which is also a part of the box lid. The robot needs to actuate this joint to fully open the lid.

Another example:
Example Input:
```Refrigerator articulation tree
links: 
base
link_0
link_1
link_2

joints: 
joint_name: joint_0 joint_type: fixed parent_link: base child_link: link_0
joint_name: joint_1 joint_type: revolute parent_link: link_0 child_link: link_1
joint_name: joint_2 joint_type: revolute parent_link: link_0 child_link: link_2

```
```Refrigerator semantics
link_0 heavy refrigerator_body
link_1 hinge door
link_2 hinge door
```

Example Output:
Task name: Close the refrigerator door
Explanation: An open fridge door is unsanitary because food can spoil due to exposure to warm air.(Unsanitary)
Description: The robotic arm will close the refrigerator door.
Additional Objects: None
Links:
- link_1: From the semantics, this is one of the refrigerator doors. The robot needs to approach this link in order to close it.
- link_2: This is the other refrigerator door. Depending on the design of the refrigerator, the robot may also need to interact with this door to fully close the refrigerator.
Joints: 
- joint_1: From the articulation tree, this is the revolute joint that connects link_1 (the first door). Therefore, the robot needs to actuate this joint to close the door.
- joint_2: This is the revolute joint that connects link_2 (the second door). Depending on the design of the refrigerator, the robot may also need to actuate this joint to fully close the refrigerator.

One more example:
Example Input:
```TrashCan articulation tree
links: 
base
link_0
link_1
link_2
link_3

joints: 
joint_name: joint_0 joint_type: continuous parent_link: link_3 child_link: link_0
joint_name: joint_1 joint_type: continuous parent_link: link_3 child_link: link_1
joint_name: joint_2 joint_type: revolute parent_link: link_3 child_link: link_2
joint_name: joint_3 joint_type: fixed parent_link: base child_link: link_3

```
```TrashCan semantics
link_0 hinge wheel
link_1 hinge wheel
link_2 hinge lid
link_3 free trashcan_body
```

Example Output:
Task name: Throw away the banana peel
Explanation: A banana peel on the floor is not safe for children because it can cause them to slip and fall.(Not safe for children)
Description: The robot arm opens the trash can lid first, then picks up the banana peel, throws it into the trash can, and finally closes the lid.
Additional Objects: Banana peel
Links:
- link_2: Link_2 is the trash can lid from the semantics. The robot needs to open and close this lid in order to throw away the banana peel.
Joints:
- joint_2: From the articulation tree, joint_2 is the revolute joint that connects link_2 (the lid). The robot needs to actuate this joint to open and close the trash can lid.

one more example:
Example Input:
```Stapler articulation tree
links: 
base
link_0
link_1
link_2

joints: 
joint_name: joint_0 joint_type: revolute parent_link: link_1 child_link: link_0
joint_name: joint_1 joint_type: revolute parent_link: link_2 child_link: link_1
joint_name: joint_2 joint_type: fixed parent_link: base child_link: link_2

```
```Stapler semantics
link_0 hinge stapler_body
link_1 hinge lid
link_2 free stapler_base

```

Example Output:
Task name: Move the stapler to the center of the table 
Explanation: A stapler left on the edge of a table is a household hazard because it can fall and injure someone, especially children. (Unsafe)
Description: The robot arm picks up the stapler and moves it to the center of the table.
Additional Objects: None
Links:
- link_0: Link_0 is the stapler body from the semantics. The robot needs to grasp this part in order to move the stapler.
- link_2: Link_2 is the stapler base. The robot may also need to interact with this part to ensure a secure grip on the stapler.
Joints:
- joint_0: From the articulation tree, joint_0 is the revolute joint that connects link_0 (the stapler body). The robot needs to actuate this joint to adjust the orientation of the stapler for a secure grip.
- joint_2: Joint_2 is the fixed joint that connects link_2 (the stapler base). The robot may need to consider the position of this joint when planning the grip and the movement.

Can you do the same for the following task and object:
"""
]

def parse_response(task_response): # 第二处改动 + 1
    task_response = '\n'.join([line for line in task_response.split('\n') if line.strip()])
    task_response = task_response.split('\n')
    task_description = None
    task_explanation = None
    additional_objects = None
    links = None
    joints = None
    task_name = None
    for l_idx, line in enumerate(task_response):
        if line.lower().startswith("task name:"):
            task_name = line.split(":")[1].strip()
            task_name = task_name.replace("/", " or ").replace(".", "").replace("'", "").replace('"', "")
            task_description = task_response[l_idx+2].split(":")[1].strip()
            task_description = task_description.replace("/", " or ").replace(".", "").replace("'", "").replace('"', "").replace(")", "").replace("(", "")
            # add explanation
            task_explanation = task_response[l_idx+1].split(":")[1].strip()
            task_explanation = task_explanation.replace("/", " or ")
            additional_objects = task_response[l_idx+3].split(":")[1].strip()
            involved_links = ""
            for link_idx in range(l_idx+5, len(task_response)):
                if task_response[link_idx].lower().startswith("joints:"):
                    break
                else:
                    involved_links += (task_response[link_idx][2:])
            links = involved_links
            involved_joints = ""
            for joint_idx in range(link_idx+1, len(task_response)):
                if not task_response[joint_idx].lower().startswith("- "):
                    break
                else:
                    involved_joints += (task_response[joint_idx][2:])
            joints = involved_joints
            break

    return task_name, task_description, task_explanation, additional_objects, links, joints

# 第三处改动
def expand_task_name(object_category, object_path, meta_path="generated_task_from_description", temperate=0, model="gpt-4"):
    ts = time.time()
    time_string = datetime.datetime.fromtimestamp(ts).strftime('%m-%d-%H-%M-%S')
    save_folder = "data/{}/{}_{}".format(meta_path, time_string, object_category)
    if not os.path.exists(save_folder + "/gpt_response"):
        os.makedirs(save_folder + "/gpt_response")
    
    save_path = "{}/gpt_response/task_generation.json".format(save_folder)

    articulation_tree_path = f"data/dataset/{object_path}/link_and_joint.txt"
    with open(articulation_tree_path, 'r') as f:
        articulation_tree = f.readlines()
    
    semantics = f"data/dataset/{object_path}/semantics.txt"
    with open(semantics, 'r') as f:
        semantics = f.readlines()

    task_user_contents_filled = copy.deepcopy(user_contents[0])
    articulation_tree_filled = """
```{} articulation tree
{}
```""".format(object_category, "".join(articulation_tree))
    semantics_filled = """
```{} semantics
{}
```""".format(object_category, "".join(semantics))
    task_user_contents_filled = task_user_contents_filled + articulation_tree_filled + semantics_filled


    system = "You are a helpful assistant."
    task_response = query(system, [task_user_contents_filled], [], save_path=save_path, debug=False, temperature=0, model=model)

    ### parse the response
    task_name, task_description, task_explanation, additional_objects, links, joints = parse_response(task_response)     # I add the explain line, So It's necessary to change the parse_response

    return task_name, task_description, task_explanation, additional_objects, links, joints, save_folder, articulation_tree_filled, semantics_filled
# 第四处改动
def generate_from_task_name(object_category, object_path, temperature_dict=None, model_dict=None, meta_path="generated_task_from_description"):
    expansion_model = model_dict.get("expansion", "gpt-4")
    expansion_temperature = temperature_dict.get("expansion", 0)
    task_name, task_description, task_explanation, additional_objects, links, joints, save_folder, articulation_tree_filled, semantics_filled = expand_task_name(
        object_category, object_path, meta_path, temperate=expansion_temperature, model=expansion_model)
    config_path = build_task_given_text(object_category, task_name, task_description, task_explanation, additional_objects, links, joints, 
                          articulation_tree_filled, semantics_filled, object_path, save_folder, temperature_dict, model_dict)
    return config_path
    
if __name__ == "__main__":
    import argparse
    import numpy as np
    from objaverse_utils.utils import partnet_mobility_dict
    # 第五处改动
    parser = argparse.ArgumentParser()
    parser.add_argument('--object', type=str, default=None)
    parser.add_argument('--object_path', type=str, default=None)
    args = parser.parse_args()
    
    temperature_dict = {
        "reward": 0.2,
        "yaml": 0.3,
        "size": 0.1,
        "joint": 0,
        "spatial_relationship": 0,
        "expansion": 0.6
    }
    
    model_dict = {
        "reward": "gpt-4",
        "yaml": "gpt-4",
        "size": "gpt-4",
        "joint": "gpt-4",
        "spatial_relationship": "gpt-4",
    }

    meta_path = "generated_task_from_description"
    # assert args.object in partnet_mobility_dict.keys(), "You should use articulated objects in the PartNet Mobility dataset."
    if args.object is None:
        args.object = partnet_categories[np.random.randint(len(partnet_categories))]
    if args.object_path is None:
        possible_object_ids = partnet_mobility_dict[args.object]
        args.object_path = possible_object_ids[np.random.randint(len(possible_object_ids))]
    config_path = generate_from_task_name(args.object, args.object_path, 
        temperature_dict=temperature_dict, meta_path=meta_path, model_dict=model_dict)
    generate_distractor(config_path, temperature_dict=temperature_dict, model_dict=model_dict)
    os.system("python execute.py --task_config_path {}".format(config_path))
    