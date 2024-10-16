import csv
from gpt_4.query import query
from gpt_4.prompts.utils import build_task_given_text
from gpt_4.prompts.prompt_distractor import generate_distractor
import time, datetime, os, copy
from manipulation.partnet_category import partnet_categories

import random

roles = {
    "Homemaker": "Responsible for managing household chores and daily life, acting as the heart of the home. Skills include expert cooking, time management, and budget control. The challenge lies in providing the best quality of life on a limited budget.",
    "Robotics Engineer": "Specializes in designing and maintaining home-use robots such as cleaning robots or elder care robots. Skills in programming, mechanical design, and AI. The challenge is developing robots that integrate seamlessly into the home environment.",
    "Gardener": "In charge of designing and maintaining the home garden. Knowledge in botany, creative design, and ecological maintenance. The challenge is to create an aesthetically pleasing yet sustainable outdoor space.",
    "Nutritionist": "Provides dietary advice and plans for family members. Expertise in nutrition, food science, and health promotion. The challenge is to balance various dietary restrictions and preferences.",
    "Personal Trainer": "Responsible for physical training and health management of family members. Skills in sports science, human physiology, and motivational psychology. The challenge is to create personalized fitness programs that accommodate varying fitness levels.",
    "Financial Planner": "Manages family finances, providing investment and savings advice. Knowledge in economics, market analysis, and risk management. The challenge is ensuring financial security and future growth for the family.",
    "Educational Consultant": "Supports children in the family with academic guidance and educational planning. Expertise in pedagogy, psychology, and curriculum design. The challenge is to adapt to different learning styles and educational needs.",
    "Home Security Officer": "Responsible for family safety and handling emergencies. Skills in security management, emergency response, and physical defense. The challenge is to maintain security without compromising the family's freedom and comfort.",
    "Interior Designer": "Optimizes the layout and design of the home to enhance living experience. Skills in artistic design, spatial planning, and color theory. The challenge is to create a functional and beautiful living space within budget.",
    "Household Advisor": "Provides comprehensive home management services from daily cleaning to organizing special events. Skills in project management, customer service, and efficiency optimization. The challenge is ensuring all household activities run efficiently and seamlessly."
}

# Randomly select three unique roles
selected_roles = random.sample(list(roles.keys()), 9)
user_profile_1 = f"You are {selected_roles[0]}. And your description: {roles[selected_roles[0]]}"
user_profile_2 = f"You are {selected_roles[1]}. And your description: {roles[selected_roles[1]]}"
user_profile_3 = f"You are {selected_roles[2]}. And your description: {roles[selected_roles[2]]}"
user_profile_4 = f"You are {selected_roles[3]}. And your description: {roles[selected_roles[3]]}"
user_profile_5 = f"You are {selected_roles[4]}. And your description: {roles[selected_roles[4]]}"
user_profile_6 = f"You are {selected_roles[5]}. And your description: {roles[selected_roles[5]]}"
user_profile_7 = f"You are {selected_roles[6]}. And your description: {roles[selected_roles[6]]}"
user_profile_8 = f"You are {selected_roles[7]}. And your description: {roles[selected_roles[7]]}"
user_profile_9 = f"You are {selected_roles[8]}. And your description: {roles[selected_roles[8]]}"



user_content_prompt = [
"""
I will provide an articulated object, with its articulation tree and semantics. Your goal is to imagine some dangerous or unsanitary household anomalies that a robotic arm can address with the articulated object. You can think of the robotic arm as a Franka Panda robot. The scenario will be built in a simulator for the robot to learn it. 
Please note that the robot arm has limited capabilities; for example, it only has one arm, making tasks like retracting the blade of a folding knife quite challenging. When setting up tasks to mitigate dangers caused by a folding knife, use simpler methods that are easier for the robot arm to handle, for example, placing the knife in a safe drawer rather than retracting the blade.

Only the articulated object sometimes will have functions, e.g., a microwave can be used to heat food, in these cases, feel free to include other objects that are needed for the task. 
Please do not think of tasks that try to assemble or disassemble the object. Do not think of tasks that aim to clean the object or check its functionality. 
The tasks will be categorized into three kinds.You just need to combine the articulated object I provided and choose a task type to write the task.
Household Hazards:This category focuses on tasks that eliminate hazards in the home, for example, put away sharp objects in the home, move objects near the edge of the table back to the center and pick up anything that might trip people.
Hygiene Management:This category focuses on tasks that ensure a clean and hygienic environment in the home, for example, dealing with spoiled food or cleaning, close the open refrigerator door to prevent food from spoiling and pick up the trash on the floor.
Child Safety Measures:This category focuses on tasks that remove dangerous objects or substances that could harm children, for example, put away medications and similar items and keep knives and sharp objects out of children's reach.
Please focus on addressing specific problems that arise in these areas.
Also you are in a group brainstorm with other teammates; as a result, answer as diversely and creatively as you can



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

Example output:
Task name: Put a knife into the box
Description: The robot arm picks up a knife and places it inside the box, then closes the box.
Explanation: A knife on the floor is dangerous because it can cause injury if someone steps on it.(Unsafe)
Additional Objects: Knife
Links:
- link_0: Link_0 is the box lid from the semantics. The robot needs to open the lid in order to put the knife inside the box.
- link_1: Link_1 is also a part of the box lid. The robot needs to ensure this part is also opened to avoid any obstruction while placing the knife inside.
Joints:
- joint_0: From the articulation tree, this is the revolute joint that connects link_0. The robot needs to actuate this joint to open and close the lid.
- joint_1: From the articulation tree, joint_1 connects link_1, which is also a part of the box lid. The robot needs to actuate this joint to fully open the lid.

Another example:
Input:
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
Output:
Task name: Throw away the banana peel
Explanation: A banana peel on the floor is not safe for children because it can cause them to slip and fall.(Not safe for children)
Description: The robot arm picks up a banana peel, opens the trash can lid, throws the banana peel into the trash can, and then closes the lid.
Additional Objects: Banana peel
Links:
- link_2: Link_2 is the trash can lid from the semantics. The robot needs to open and close this lid in order to throw away the banana peel.
Joints:
- joint_2: From the articulation tree, joint_2 is the revolute joint that connects link_2 (the lid). The robot needs to actuate this joint to open and close the trash can lid.

Can you do the same for the following task and object:


"""
]

# Initialize user_contents for each role
user_contents_1 = user_profile_1 + "\n" + user_content_prompt[0]
user_contents_2 = user_profile_2 + "\n" + user_content_prompt[0]
user_contents_3 = user_profile_3 + "\n" + user_content_prompt[0]
user_contents_4 = user_profile_4 + "\n" + user_content_prompt[0]
user_contents_5 = user_profile_5 + "\n" + user_content_prompt[0]
user_contents_6 = user_profile_6 + "\n" + user_content_prompt[0]
user_contents_7 = user_profile_7 + "\n" + user_content_prompt[0]
user_contents_8 = user_profile_8 + "\n" + user_content_prompt[0]
user_contents_9 = user_profile_9 + "\n" + user_content_prompt[0]

user_contents = [
    user_contents_1,
    user_contents_2,
    user_contents_3,
    user_contents_4,
    user_contents_5,
    user_contents_6,
    user_contents_7,
    user_contents_8,
    user_contents_9
]

def parse_response(task_response):
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
            task_name = line.split(":", 1)[1].strip()
            task_name = task_name.replace("/", " or ").replace(".", "").replace("'", "").replace('"', "")
            task_description = task_response[l_idx+2].split(":", 1)[1].strip()
            task_description = task_description.replace("/", " or ").replace(".", "").replace("'", "").replace('"', "").replace(")", "").replace("(", "")
            task_explanation = task_response[l_idx+1].split(":", 1)[1].strip()
            task_explanation = task_explanation.replace("/", " or ")
            additional_objects = task_response[l_idx+3].split(":", 1)[1].strip()
            involved_links = ""
            for link_idx in range(l_idx+5, len(task_response)):
                if task_response[link_idx].lower().startswith("joints:"):
                    break
                else:
                    involved_links += (task_response[link_idx][2:] + "\n")
            links = involved_links.strip()
            involved_joints = ""
            for joint_idx in range(link_idx+1, len(task_response)):
                if not task_response[joint_idx].lower().startswith("- "):
                    break
                else:
                    involved_joints += (task_response[joint_idx][2:] + "\n")
            joints = involved_joints.strip()
            break

    return task_name, task_description, task_explanation, additional_objects, links, joints

def expand_task_name(object_category, object_path, round, time_string, meta_path="generated_task", temperate=0, model="gpt-4"):
    save_folder = "data/{}/{}_{}".format(meta_path, time_string, object_category)
    if not os.path.exists(save_folder + "/gpt_response"):
        os.makedirs(save_folder + "/gpt_response")
    
    save_path_1 = "{}/gpt_response/task_generation{}.json".format(save_folder, round*3-2)
    save_path_2 = "{}/gpt_response/task_generation{}.json".format(save_folder, round*3-1)
    save_path_3 = "{}/gpt_response/task_generation{}.json".format(save_folder, round*3)


    articulation_tree_path = f"data/dataset/{object_path}/link_and_joint.txt"
    with open(articulation_tree_path, 'r') as f:
        articulation_tree = f.readlines()
    
    semantics = f"data/dataset/{object_path}/semantics.txt"
    with open(semantics, 'r') as f:
        semantics = f.readlines()

    articulation_tree_filled = """
```{} articulation tree
{}
```""".format(object_category, "".join(articulation_tree))
    semantics_filled = """
```{} semantics
{}
```""".format(object_category, "".join(semantics))

    system = "You are a helpful assistant."
    task_response_1 = query(system, [user_contents[round*3-3] + articulation_tree_filled + semantics_filled], [], save_path=save_path_1, debug=False, temperature=temperate, model=model)
    task_response_2 = query(system, [user_contents[round*3-2] + articulation_tree_filled + semantics_filled], [], save_path=save_path_2, debug=False, temperature=temperate, model=model)
    task_response_3 = query(system, [user_contents[round*3-1] + articulation_tree_filled + semantics_filled], [], save_path=save_path_3, debug=False, temperature=temperate, model=model)

    task_name_1, task_description_1, task_explanation_1, additional_objects_1, links_1, joints_1 = parse_response(task_response_1)
    task_name_2, task_description_2, task_explanation_2, additional_objects_2, links_2, joints_2 = parse_response(task_response_2)
    task_name_3, task_description_3, task_explanation_3, additional_objects_3, links_3, joints_3 = parse_response(task_response_3)
    return task_name_1, task_description_1, task_explanation_1, additional_objects_1, links_1, joints_1, \
           task_name_2, task_description_2, task_explanation_2, additional_objects_2, links_2, joints_2, \
           task_name_3, task_description_3, task_explanation_3, additional_objects_3, links_3, joints_3, \
           save_folder, articulation_tree_filled, semantics_filled

def generate_from_task_name(object_category, object_path, temperature_dict=None, model_dict=None, meta_path="generated_task"):
    expansion_model = model_dict.get("expansion", "gpt-4")
    expansion_temperature = temperature_dict.get("expansion", 0.6)
    task_name, task_description, task_explanation, additional_objects, links, joints, \
    save_folder, articulation_tree_filled, semantics_filled = expand_task_name(
        object_category, object_path, meta_path, temperate=expansion_temperature, model=expansion_model)
    config_path = build_task_given_text(object_category, task_name, task_description, task_explanation, additional_objects, links, joints, 
                          articulation_tree_filled, semantics_filled, object_path, save_folder, temperature_dict, model_dict)
    return config_path

def brainstorming_loop(object_category, object_path, num_rounds=3, meta_path="generated_task", temperature=0, model="gpt-4"):
    previous_tasks = []
    csv_data = []
    ts = time.time()
    time_string = datetime.datetime.fromtimestamp(ts).strftime('%m-%d-%H-%M-%S')
    for round in range(1, num_rounds + 1):
        # Generate tasks
        task_name_1, task_description_1, task_explanation_1, additional_objects_1, links_1, joints_1, \
        task_name_2, task_description_2, task_explanation_2, additional_objects_2, links_2, joints_2, \
        task_name_3, task_description_3, task_explanation_3, additional_objects_3, links_3, joints_3, \
        save_folder, articulation_tree_filled, semantics_filled = expand_task_name(
            object_category, object_path, round, time_string, meta_path, temperate=temperature, model=model)
        
        # Collect tasks
        new_tasks = [
            (task_name_1, task_description_1, task_explanation_1, additional_objects_1, links_1, joints_1),
            (task_name_2, task_description_2, task_explanation_2, additional_objects_2, links_2, joints_2),
            (task_name_3, task_description_3, task_explanation_3, additional_objects_3, links_3, joints_3)
        ]
        previous_tasks.extend(new_tasks)
        
        # Add data to csv_data
        csv_data.extend([
            [task_name_1, task_description_1,  selected_roles[round*3-3], round],
            [task_name_2, task_description_2,  selected_roles[round*3-2], round],
            [task_name_3, task_description_3,  selected_roles[round*3-1], round]
        ])
        
        # Update user contents for next round
        update_user_contents(previous_tasks, round)
        
        print(f"Completed round {round} of {num_rounds}")
    
    # Save results to CSV
    csv_path = f"{save_folder}/brainstorming_results.csv"
    
    # Check if the file already exists
    file_exists = os.path.isfile(csv_path)
    
    with open(csv_path, 'a', newline='', encoding='utf-8') as csvfile:
        csv_writer = csv.writer(csvfile)
        
        # Write header only if the file is newly created
        if not file_exists:
            csv_writer.writerow(['Task Name', 'Task Description', 'Task Explanation', 'Additional Objects', 'Links', 'Joints', 'Agent Role', 'Round'])
        
        # Write the new data
        csv_writer.writerows(csv_data)
    
    print(f"Results appended to {csv_path}")
    
    return previous_tasks, save_folder, articulation_tree_filled, semantics_filled

def update_user_contents(previous_tasks, round):
    global user_contents_1, user_contents_2, user_contents_3, user_contents_4, user_contents_5, user_contents_6, user_contents_7, user_contents_8, user_contents_9

    task_info = "\n\nHere are the previously generated tasks for your reference:\n"
    for task in previous_tasks:
        task_info += f"Task name: {task[0]}\nDescription: {task[1]}\nExplanation: {task[2]}\nAdditional Objects: {task[3]}\n\n"

    globals()[f"user_contents_{round*3-2}"] = globals()[f"user_profile_{round*3-2}"] + task_info + "\n" + user_content_prompt[0]
    globals()[f"user_contents_{round*3-1}"] = globals()[f"user_profile_{round*3-1}"] + task_info + "\n" + user_content_prompt[0]
    globals()[f"user_contents_{round*3}"] = globals()[f"user_profile_{round*3}"] + task_info + "\n" + user_content_prompt[0]

if __name__ == "__main__":
    import argparse
    import numpy as np
    from objaverse_utils.utils import partnet_mobility_dict

    parser = argparse.ArgumentParser()
    parser.add_argument('--object', type=str, default=None)
    parser.add_argument('--object_path', type=str, default=None)
    parser.add_argument('--num_rounds', type=int, default=3)
    args = parser.parse_args()
    
    temperature_dict = {
        "reward": 0.2,
        "yaml": 0.3,
        "size": 0.1,
        "joint": 0,
        "spatial_relationship": 0,
        "expansion": 0.7
    }
    
    model_dict = {
        "reward": "gpt-4",
        "yaml": "gpt-4",
        "size": "gpt-4",
        "joint": "gpt-4",
        "spatial_relationship": "gpt-4",
        "expansion": "gpt-4"
    }

    meta_path = "generated_task"
    if args.object is None:
        args.object = partnet_categories[np.random.randint(len(partnet_categories))]
    if args.object_path is None:
        possible_object_ids = partnet_mobility_dict[args.object]
        args.object_path = possible_object_ids[np.random.randint(len(possible_object_ids))]
    
    # Start the brainstorming loop
    all_tasks, save_folder, articulation_tree_filled, semantics_filled = brainstorming_loop(
        args.object, args.object_path, args.num_rounds, temperature=temperature_dict["expansion"], model=model_dict["expansion"])
    
    # Process the generated tasks
    for task in all_tasks:
        task_name, task_description, task_explanation, additional_objects, links, joints = task
        config_path = build_task_given_text(
            args.object, task_name, task_description, task_explanation, additional_objects, links, joints,
            articulation_tree_filled, semantics_filled, args.object_path, save_folder, temperature_dict, model_dict)
        generate_distractor(config_path, temperature_dict=temperature_dict, model_dict=model_dict)
