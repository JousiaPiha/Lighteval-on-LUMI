#!/bin/bash -l
#SBATCH --job-name=poro34b-le
#SBATCH --output=poro34b-le.o%j
#SBATCH --error=poro34b-le.e%j
#SBATCH --partition=standard-g
#SBATCH --nodes=1
#SBATCH --gres=gpu:mi250:2
#SBATCH --time=1-0:00:00
#SBATCH --account=project_<PROJECT_ID>

PROJECT_FOLDER="project_<PROJECT_ID>"
PROJECT_PATH="/projappl/$PROJECT_FOLDER/$USER/Lighteval-on-LUMI"
module use /appl/local/csc/modulefiles/
module purge
module load LUMI
module load pytorch nano
export HF_HOME=/scratch/$PROJECT_FOLDER/hf_cache
export PYTHONPATH="$PROJECT_PATH/.venv/lib/python3.10/site-packages/"
source $PROJECT_PATH/.venv/bin/activate

srun singularity_wrapper exec accelerate launch --num_machines=1 \
  --num_processes=1 \
  --mixed_precision="bf16" \
  $PROJECT_PATH/run_evals_accelerate.py \
  --model_args="pretrained=LumiOpen/Poro-34B,max_length=2048,model_parallel=True,dtype=bfloat16" \
  --tasks "leaderboard|arc:challenge|25|0,leaderboard|hellaswag|10|0,original|mmlu|5|0,leaderboard|truthfulqa:mc|0|0,leaderboard|winogrande|5|0" \
  --output_dir="$PROJECT_PATH/evals/poro/" \
  --override_batch_size=4
