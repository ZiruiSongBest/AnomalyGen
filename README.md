<img src="imgs\Task.png" alt="Local image">

<h1 style="text-align: center;">AnomalyGen</h1>
## Installation
1.Clone this repository
```bash
git clone https://github.com/ZiruiSongBest/AnomalyGen.git
```
2.Set up the environment
```bash
conda env create -f environment.yaml
pip install ompl-1.6.0-cp39-cp39-manylinux_2_28_x86_64.whl
```
3.Download the Dataset  
We provide a parsed version [dataset](https://drive.google.com/file/d/1d-1txzcg_ke17NkHKAolXlfDnmPePFc6/view?usp=sharing) based on PartNet-Mobility  
Please download and unzip it in the `data` folder  

We also provide the embeddings [here](https://drive.google.com/file/d/1dFDpG3tlckTUSy7VYdfkNqtfVctpn3T6/view?usp=sharing). After downloading, unzip and put it under `objaverse_utils/data/` folder
## Run
Put your OpenAI API key in the `Go.sh` file.  
1.Generate the task  
This script will initiate the brainstorming process to generate task instructions using GPT-4.

```bash
source Go.sh
python gpt_4\prompts\prompt_brainstorming.py
```

2.Excute the task  
After generating the tasks, navigate to the `data/generated_task` folder. You will see folders named in the `time_object` format, such as `10-13-22-01-31_Refrigerator`. Inside, there are two sets of YAML files. One set starts with `The_robotic_arm_will`, which contains the configuration files for distractors in the room. The other set is the execution files we need, named according to the task descriptions.  
To excute the task  

```bash
python execute.py --task_config_path The_path_to_the_Task_description_YAML_Not_begin_with_"The_robotic_arm_will"
```

Or you can run this command for a better result.

```bash
python execute_long_horizon.py --task_config_path The_path_to_the_Task_description_YAML_Not_begin_with_"The_robotic_arm_will"
```

