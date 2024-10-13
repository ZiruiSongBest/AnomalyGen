source ~/.bashrc
export OPENAI_API_KEY=""
export PYTHONPATH=${PWD}:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0,1,2,3
conda activate AnomalyGen
