import copy
import os
import yaml
from gpt_4.prompts.prompt_manipulation_reward_primitive import decompose_and_generate_reward_or_primitive
from gpt_4.prompts.prompt_set_joint_angle import query_joint_angle
from gpt_4.prompts.prompt_spatial_relationship import query_spatial_relationship
from gpt_4.query import query
from gpt_4.adjust_size import adjust_size_v2

task_yaml_config_prompt = """
I need you to describe the initial scene configuration for a given task in the following format, using a yaml file. This YAML file will help build the task in a simulator.
The format is as follows:
```yaml 
- use_table: Whether the task requires using a table should be determined based on common sense and the task's specific needs. If a table is used, its location will be fixed at (0, 0, 0). The height of the table will be 0.6m. Usually, if the objects invovled in the task are usually placed on a table (not directly on the ground), then the task requires using a table.
# for each object involved in the task, we need to specify the following fields for it.
- type: mesh
  name: name of the object, so it can be referred to in the simulator
  size: describe the scale of the object mesh using 1 number in meters. The scale should match real everyday objects. E.g., an apple is of scale 0.08m. You can think of the scale to be the longest dimension of the object. 
  lang: this should be a language description of the mesh. The language should be a concise description of the obejct, such that the language description can be used to search an existing database of objects to find the object.
  path: this can be a string showing the path to the mesh of the object. 
  on_table: whether the object needs to be placed on the table (if there is a table needed for the task). This should be based on common sense and the requirement of the task. E.g.,Objects that can trip people are usually on the floor, not on the table.(If the task is to pick up an object that could trip someone, then the object should be on the floor.)
  center: the location of the object center. If there isn't a table needed for the task or the object does not need to be on the table, this center should be expressed in the world coordinate system. If there is a table in the task and the object needs to be placed on the table, this center should be expressed in terms of the table coordinate, where (0, 0, 0) is the lower corner of the table, and (1, 1, 1) is the higher corner of the table. In either case, you should try to specify a location such that there is no collision between objects.
  movable: if the object is movable or not in the simulator due to robot actions.It should be true if the task specifically requires the robot to move the object.
```

An example input includes the task names, task descriptions, and objects involved in the task. I will also provide with you the articulation tree and semantics of the articulated object. 
This can be useful for knowing what parts are already in the articulated object, and thus you do not need to repeat those parts as separate objects in the yaml file.

Your task includes two parts:
1. Output the yaml configuration of the task.
2. Sometimes, the task description / objects involved will refer to generic/placeholder objects, e.g., to place an "item" into the drawer, and to heat "food" in the microwave. In the generated yaml config, you should change these placeholder objects to be concrete objects in the lang field, e.g., change "item" to be a toy or a pencil, and "food" to be a hamburger, a bowl of soup, etc. 

Example input:
Task Name: Put a knife into the box
Description: The robot arm opens the box first, then picks up the knife, places it inside the box, and finally closes the box.
Explanation: A knife on the floor is dangerous because it can cause injury if someone steps on it.(Unsafe)
Objects involved: Box, Knife

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


An example output:
```yaml
- use_table: True # According to the task, the box is used to safely store kitchen knives, so a table is required.
- type: mesh
  name: "Box"
  on_table: True # According to the task, the box is used to safely store kitchen knives, so it should be placed on the table.
  center: (0.3, 0.6, 0) # Remember that when an object is placed on the table, the center is expressed in the table coordinate, where (0, 0, 0) is the lower corner and (1, 1, 1) is the higher corner of the table. Here we put the box just at a random location on the table.  
  size: 0.3 # the size of a common box is roughly 0.3m
  lang: "a common box"
  path: "box.urdf"
- type: mesh
  name: "Knife"
  on_table: False # According to the task explanation, the knife should be on the floor.
  center: (0.7, 0.4, 0) # Remember that when not on a table, the center is expressed in the world coordinate. Since the robot is at (1, 1, 0) and the table is at (0, 0, 0), we place the knife at (1.8, 2, 0) to avoid collision with the table and the robot.
  size: 0.3 # common size of a knife 
  lang: "a kitchen knife"
  path: "knife.obj"
  movable: True # here the task requires the robot to move the knife, so the knife has to be moveable.
```

Another example input:
Task name: Close the refrigerator door
Description: The robotic arm will close the refrigerator door.
Explanation: An open fridge door is unsanitary because food can spoil due to exposure to warm air.(Unsanitary)
Additional Objects: None

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
Output:
```yaml
- use_table: False # According to the task description, there is no specific request. The refrigerator should simply be placed on the floor based on common sense.
- type: mesh
  name: "Refrigerator"
  on_table: False # According to the task description, there is no specific request. The refrigerator should simply be placed on the floor based on common sense.
  center: (1.5, 0.5, 0) # Remember that when not on a table, the center is expressed in the world coordinate. Since the robot is at (1, 1, 0) and the table is at (0, 0, 0), we place the refrigerator at (1.5, 0.5, 0) to avoid collision with the table and the robot.
  movable: False # The refrigerator is not movable, only its door is.
  size: 1.8 # the size of a common refrigerator is roughly 1.8m
  lang: "a common refrigerator"
  path: "refrigerator.urdf"
```
Note in this example, the refrigerator already has door and refrigerator_body from the semantics file. Therefore, you do not need to include a separate door or refrigerator_body in the yaml file.


One more example input:
Task Name: Throw away the banana peel
Description: The robot arm opens the trash can lid first, then picks up the banana peel, throws it into the trash can, and finally closes the lid.
Explanation: A banana peel on the floor is not safe for children because it can cause them to slip and fall.(Not safe for children)
Objects involved: TrashCan, Banana peel

```TrashCan articulation tree
links: 
base
link_0
link_1

joints: 
joint_name: joint_0 joint_type: fixed parent_link: base child_link: link_0
joint_name: joint_1 joint_type: prismatic parent_link: link_0 child_link: link_1

```
```TrashCan semantics
link_0 free trashcan_body
link_1 slider lid
```

Output:
```yaml
- use_table: False # Trash can and banana peel are usually on the floor.
- type: mesh
  name: "TrashCan"
  on_table: False # According to the task explanation and common sense, the trashcan should be on the floor.
  center: (1.5, 1.5, 0) # Remember that when not on a table, the center is expressed in the world coordinate. Since the robot is at (1, 1, 0) and the table is at (0, 0, 0), we place the trash can at (1.5, 1.5, 0) to avoid collision with the table and the robot.
  movable: False  # Task don't need to move the TrashCan.
  size: 0.6 # the size of a trash can is roughly 0.6m
  lang: "a common trash can"
  path: "trashcan.urdf"
- type: mesh
  name: "Banana peel"
  on_table: False # According to the task, the banana peel should be on the floor.
  center: (1.2, 1.2, 0) # Remember that when not on a table, the center is expressed in the world coordinate. Since the robot is at (1, 1, 0) and the table is at (0, 0, 0), we place the banana peel at (1.2, 1.2, 0) to avoid collision with the table and the robot.
  movable: True      # According to task, banana peel should be movable to be thrown in TrashCan.
  size: 0.1 # common size of a banana peel 
  lang: "a banana peel"
  path: "banana_peel.obj"
```

Another example:
Task Name: Put an item into the box drawer
Description: The robot will open the drawer of the box, and put an item into it.
Explanation:None
Objects involved: A box with drawer, an item to be placed in the drawer. 

```Box articulation tree
links: 
base
link_0
link_1
link_2

joints: 
joint_name: joint_0 joint_type: revolute parent_link: link_2 child_link: link_0
joint_name: joint_1 joint_type: prismatic parent_link: link_2 child_link: link_1
joint_name: joint_2 joint_type: fixed parent_link: base child_link: link_2
```

```Box semantics
link_0 hinge rotation_lid
link_1 slider drawer
link_2 free box_body
```

Output:
```yaml
-   use_table: true
-   center: (0.5, 0.5, 0)
    lang: "a wooden box"
    name: "Box"
    on_table: true
    path: "box.urdf"
    size: 0.3
    type: urdf
-   path: "item.obj"
    center: (0.2, 0.4, 0)
    lang: "A toy" # Note here, we changed the generic/placeholder "item" object to be a more concrete object: a toy. 
    name: "Item"
    on_table: true
    size: 0.05
    type: mesh
```

One more example:
Task Name: Fetch item from refrigerator
Description: The robot will open the refrigerator door, and fetch an item from the refrigerator.
Explanation:None
Objects involved: A refrigerator, an item to be fetched from the refrigerator.

```Refirgerator articulation tree
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

Output:
```yaml
-   use_table: true # the fetched item should be placed on the table, after it's moved out of the refrigerator.
-   center: (1.0, 0.2, 0) # Remember that when not on a table, the center is expressed in the world coordinate. Since the robot is at (1, 1, 0) and the table is at (0, 0, 0), we place the oven at (1.8, 2, 0) to avoid collision with the table and the robot.
    lang: a common two-door refrigerator
    name: Refrigerator
    on_table: false # the refrigerator is usually placed on the floor.
    path: refrigerator.urdf
    reward_asset_path: '10612'
    size: 1.8
    type: urdf
-   center: (1.0, 0.2, 0.5) # the soda can is initially placed inside the refrigerator.
    lang: a can of soda
    name: Item
    on_table: false # the item is initially placed inside the refrigerator
    path: soda_can.obj
    size: 0.2
    type: mesh
```

Rules: 
- You do not need to include the robot in the yaml file.
- The yaml file should only include the objects listed in "Objects involved".
- Sometimes, the task description / objects involved will refer to generic/placeholder objects, e.g., to place an "item" into the drawer, and to heat "food" in the microwave. In the generated yaml config, you should change these placeholder objects to be concrete objects in the lang field, e.g., change "item" to be a toy or a pencil, and "food" to be a hamburger, a bowl of soup, etc. 


Can you do this for the following task:
Task Name: {}
Description: {}
Explanation: {}
Objects involved: {}
"""

def parse_response_to_get_yaml(response, task_description, save_path, temperature=0.2, model='gpt-4'):
    yaml_string = []
    for l_idx, line in enumerate(response):
        if "```yaml" in line:
            for l_idx_2 in range(l_idx + 1, len(response)):
                if response[l_idx_2].lstrip().startswith("```"):
                    break

                yaml_string.append(response[l_idx_2])

            yaml_string = '\n'.join(yaml_string)
            description = f"{task_description}".replace(" ", "_").replace(".", "").replace(",", "").replace("(", "").replace(")", "")
            save_name =  description + '.yaml'

            print("=" * 30)
            print("querying GPT to adjust the size of the objects")
            print("=" * 30)
            parsed_size_yaml = adjust_size_v2(description, yaml_string, save_path, temperature, model=model)

            return parsed_size_yaml, save_name

def parse_task_response(task_response):
    task_names = []
    task_descriptions = []
    additional_objects = []
    links = []
    joints = []

    task_response = task_response.split("\n")
    for l_idx, line in enumerate(task_response):
        if line.lower().startswith("task name:"):
            task_name = line.split(":")[1].strip()
            task_name = task_name.replace("/", " or ").replace(".", "").replace("'", "").replace('"', "")
            task_names.append(task_name)
            task_description = task_response[l_idx+1].split(":")[1].strip()
            task_description = task_description.replace("/", " or ").replace(".", "").replace("'", "").replace('"', "").replace(")", ".").replace("(", ".")
            task_descriptions.append(task_description)
            additional_objects.append(task_response[l_idx+2].split(":")[1].strip())
            involved_links = ""
            for link_idx in range(l_idx+4, len(task_response)):
                if task_response[link_idx].lower().startswith("joints:"):
                    break
                else:
                    # involved_links.append(task_response[link_idx].split(":")[0][2:])
                    involved_links += (task_response[link_idx][2:])
            links.append(involved_links)
            involved_joints = ""
            for joint_idx in range(link_idx+1, len(task_response)):
                if not task_response[joint_idx].lower().startswith("- "):
                    break
                else:
                    # involved_joints.append(task_response[joint_idx].split(":")[0][2:])
                    involved_joints += (task_response[joint_idx][2:])
            joints.append(involved_joints)

    return task_names, task_descriptions, additional_objects, links, joints

def build_task_given_text(object_category, task_name, task_description, task_explanation, additional_object, involved_links, involved_joints, 
                          articulation_tree_filled, semantics_filled, object_path, save_folder, temperature_dict, model_dict=None):
    if model_dict is None:
        model_dict = {
            "task_generation": "gpt-4",
            "reward": "gpt-4",
            "yaml": "gpt-4",
            "size": "gpt-4",
            "joint": "gpt-4",
            "spatial_relationship": "gpt-4"
        }

    task_yaml_config_prompt_filled = copy.deepcopy(task_yaml_config_prompt)
    if additional_object.lower() == "none":
        task_object = object_category
    else:
        task_object = "{}, {}".format(object_category, additional_object)
    task_yaml_config_prompt_filled = task_yaml_config_prompt_filled.format(task_name, task_description, task_explanation, task_object)
    task_yaml_config_prompt_filled += articulation_tree_filled + semantics_filled

    system = "You are a helpful assistant."
    save_path = os.path.join(save_folder, "gpt_response/task_yaml_config_{}.json".format(task_name))
    print("=" * 50)
    print("=" * 20, "generating task yaml config", "=" * 20)
    print("=" * 50)
    task_yaml_response = query(system, [task_yaml_config_prompt_filled], [], save_path=save_path, debug=False, 
                            temperature=temperature_dict["yaml"], model=model_dict["yaml"])
    # NOTE: parse the yaml file and generate the task in the simulator.
    description = f"{task_name}_{task_description}".replace(" ", "_").replace(".", "").replace(",", "")
    task_yaml_response = task_yaml_response.split("\n")
    size_save_path = os.path.join(save_folder, "gpt_response/size_{}.json".format(task_name))
    parsed_yaml, save_name = parse_response_to_get_yaml(task_yaml_response, description, save_path=size_save_path, 
                                                        temperature=temperature_dict["size"], model=model_dict["size"])

    # NOTE: post-process such that articulated object is urdf.
    # NOTE: post-process to include the reward asset path for reward generation. 
    for obj in parsed_yaml:
        if "name" in obj and obj['name'] == object_category:
            obj['type'] = 'urdf'
            obj['reward_asset_path'] = object_path

    # config_path = "gpt_4/data/parsed_configs_semantic_articulated/{}-{}".format(object_category, time_string)
    config_path = save_folder
    with open(os.path.join(config_path, save_name), 'w') as f:
        yaml.dump(parsed_yaml, f, indent=4)

    input_to_reward_config = copy.deepcopy(parsed_yaml)
    for obj in input_to_reward_config:
        if "reward_asset_path" in obj:
            input_to_reward_config.remove(obj)
    initial_config = yaml.safe_dump(parsed_yaml)

    ### decompose and generate reward
    yaml_file_path = os.path.join(config_path, save_name)
    reward_save_path = os.path.join(save_folder, "gpt_response/reward_{}.json".format(task_name))
    print("=" * 50)
    print("=" * 20, "generating reward", "=" * 20)
    print("=" * 50)
    solution_path = decompose_and_generate_reward_or_primitive(task_name, task_description, initial_config, 
                                                                articulation_tree_filled, semantics_filled, 
                                                                involved_links, involved_joints, object_path, 
                                                                yaml_file_path, save_path=reward_save_path,
                                                                temperature=temperature_dict["reward"],
                                                                model=model_dict["reward"])
    

    ### generate joint angle
    save_path = os.path.join(save_folder, "gpt_response/joint_angle_{}.json".format(task_name))
    substep_file_path = os.path.join(solution_path, "substeps.txt")
    with open(substep_file_path, 'r') as f:
        substeps = f.readlines()
    print("=" * 50)
    print("=" * 20, "generating initial joint angle", "=" * 20)
    print("=" * 50)
    joint_angle_values = query_joint_angle(task_name, task_description, articulation_tree_filled, semantics_filled, 
                                            involved_links, involved_joints, substeps, save_path=save_path, 
                                            temperature=temperature_dict['joint'], model=model_dict["joint"])
    joint_angle_values["set_joint_angle_object_name"] = object_category

    involved_objects = []
    config = yaml.safe_load(initial_config)
    for obj in config:
        if "name" in obj:
            involved_objects.append(obj["name"])
    involved_objects = ", ".join(involved_objects)
    save_path = os.path.join(save_folder, "gpt_response/spatial_relationships_{}.json".format(task_name))
    print("=" * 50)
    print("=" * 20, "generating initial spatial relationship", "=" * 20)
    print("=" * 50)
    spatial_relationships = query_spatial_relationship(task_name, task_description, involved_objects, articulation_tree_filled, semantics_filled, 
                                            involved_links, involved_joints, substeps, save_path=save_path, 
                                            temperature=temperature_dict['spatial_relationship'], model=model_dict["spatial_relationship"])

    config.append(dict(solution_path=solution_path))
    config.append(joint_angle_values)
    config.append(dict(spatial_relationships=spatial_relationships))
    config.append(dict(task_name=task_name, task_description=task_description))
    with open(os.path.join(config_path, save_name), 'w') as f:
        yaml.dump(config, f, indent=4)
    with open(os.path.join(solution_path, "config.yaml"), 'w') as f:
        yaml.dump(config, f, indent=4)

    return os.path.join(config_path, save_name)