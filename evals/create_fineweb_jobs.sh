# This script is for running Fineweb-specific evaluation jobs with Lighteval on LUMI.
# The script will interate through a model folder containing checkpoint folders and create a SLURM job for each checkpoint folder.
# The script will make sure that the checkpoint folder contains a model checkpoint file, a config file and tokenizer.json file.
# The script takes in the following arguments:
    # 1. model_folder: The folder containing the model checkpoint folders. Required.
    # 2. job_name_prefix: The prefix to use for the job name. Defaults to "LEFW-".
    # 3. partition: The partition to use for the job. Defaults to "standard-g".
    # 4. time: The time to allocate for the job. Defaults to "0:20:00".
    # 5. gpu: The number of GPUs to allocate for the job. Defaults to "1".
    # 6. account: The account to use for the job. Required.
    # 7. eval_output: The folder to save the evaluation output. Defaults to "./evals/"
    # 8. log_output: The folder to save the job logs. Defaults to "./logs/"
    # 9. jobs_output: The folder to save the job scripts. Defaults to "./jobs/"
    # 10. model_dtype: The data type of the model. Defaults to "bfloat16".
    # 11. virtual_env: The virtual environment to use for the job. Defaults to "./.venv".

# Example usage:
# bash screate_jobs.sh /path/to/model/folder/ job_name_prefix partition time gpu account eval_output log_output jobs_output model_dtype
module use /appl/local/csc/modulefiles/
module purge
module load LUMI
module load pytorch
set -e
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir"
# Function to show help message
usage() {
    echo "Usage: bash screate_jobs.sh --model_folder <folder_path> --account <account_name> [options]"
    echo "Required arguments:"
    echo "  --model_folder   Path to the model folder"
    echo "  --account        Account name"
    echo "Optional arguments:"
    echo "  --job_name_prefix Prefix for job names (default: LEFW-)"
    echo "  --partition       Partition to use (default: standard-g)"
    echo "  --time            Time for the job (default: 0:20:00)"
    echo "  --gpu             Number of GPUs (default: 1)"
    echo "  --eval_output     Evaluation output folder (default: ./evals/)"
    echo "  --log_output      Logs output folder (default: ./logs/)"
    echo "  --jobs_output     Jobs output folder (default: ./jobs/)"
    echo "  --model_dtype     Model data type (default: bfloat16)"
    echo "  --virtual_env     Path to virtual environment (default: ./.venv)"
    exit 0
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --model_folder)
        model_folder="$2"
        shift # past argument
        shift # past value
        ;;
        --account)
        account="$2"
        shift
        shift
        ;;
        --job_name_prefix)
        job_name_prefix="$2"
        shift
        shift
        ;;
        --partition)
        partition="$2"
        shift
        shift
        ;;
        --time)
        time="$2"
        shift
        shift
        ;;
        --gpu)
        gpu="$2"
        shift
        shift
        ;;
        --eval_output)
        eval_output="$2"
        shift
        shift
        ;;
        --log_output)
        log_output="$2"
        shift
        shift
        ;;
        --jobs_output)
        jobs_output="$2"
        shift
        shift
        ;;
        --model_dtype)
        model_dtype="$2"
        shift
        shift
        ;;
        --virtual_env)
        virtual_env="$2"
        shift
        shift
        ;;
        --help|-h)
        usage
        ;;
        *)
        echo "Unknown option: $1"
        usage
        ;;
    esac
done

# Ensure required arguments are provided
if [ -z "$model_folder" ] || [ -z "$account" ]; then
    echo "Error: --model_folder and --account are required."
    usage
fi

# Check if model_folder exists
if [ ! -d "$model_folder" ]; then
    echo "Error: Model folder '$model_folder' does not exist."
    exit 1
fi

# Set defaults for optional arguments
job_name_prefix=${job_name_prefix:-"LEFW-"}
partition=${partition:-"standard-g"}
time=${time:-"0:20:00"}
gpu=${gpu:-"1"}
eval_output=${eval_output:-"./evals/"}
log_output=${log_output:-"./logs/"}
jobs_output=${jobs_output:-"./jobs/"}
model_dtype=${model_dtype:-"bfloat16"}
virtual_env=${virtual_env:-"./.venv"}

echo "Model folder: $model_folder"
echo "Account: $account"
echo "Job name prefix: $job_name_prefix"
echo "Partition: $partition"
echo "Time: $time"
echo "GPU: $gpu"
echo "Eval output: $eval_output"
echo "Log output: $log_output"
echo "Jobs output: $jobs_output"
echo "Model dtype: $model_dtype"
echo "Virtual env: $virtual_env"


# Check if the virtual environment exists
echo "Checking if the virtual environment exists."
if [ ! -d "$virtual_env" ]; then
    echo "Virtual environment does not exist. Creating it with --system-site-packages."
    python -m venv --system-site-packages "$virtual_env"
fi
# Check Python version
echo "Checking if Python 3.10 is the active version in the virtual environment."
python_version=$($virtual_env/bin/python -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
if [ "$python_version" != "3.10" ]; then
    echo "Python 3.10 is not the active version in the virtual environment."
    exit 1
fi
# Activate the virtual environment
echo "Activating the virtual environment."
source "$virtual_env/bin/activate"

# Correct PYTHONPATH
echo "Correcting PYTHONPATH."
unset PYTHONPATH
export PYTHONPATH="$virtual_env/lib/python3.10/site-packages"

# Check if the virtual environment contains the required packages
echo "Checking if the virtual environment contains the required packages."
if ! $virtual_env/bin/python -c "import lighteval, accelerate" 1> /dev/null 2>&1; then
    echo "Lighteval is not installed in the virtual environment. Installing it."
    pip install lighteval[accelerate,vllm,extended_tasks]
fi

# Create folders if they do not exist
echo "Creating folders if they do not exist."
mkdir -p "$eval_output"
mkdir -p "$log_output"
mkdir -p "$jobs_output"

# Check if custom task file lighteval_tasks.py exists in ./evals/tasks/ and if not, download it
echo "Checking if custom task file lighteval_tasks.py exists in ./evals/tasks/ and if not, downloading it."
if [ ! -f "./evals/tasks/lighteval_tasks.py" ]; then
    mkdir -p "./evals/tasks/"
    wget -O "./evals/tasks/lighteval_tasks.py" "https://raw.githubusercontent.com/JousiaPiha/Lighteval-on-LUMI/main/evals/tasks/lighteval_tasks.py"
fi
# Check if task list file fineweb.txt exists in ./evals/tasks/ and if not, download it
echo "Checking if task list file fineweb.txt exists in ./evals/tasks/ and if not, downloading it."
if [ ! -f "./evals/tasks/fineweb.txt" ]; then
    wget -O "./evals/tasks/fineweb.txt" "https://raw.githubusercontent.com/JousiaPiha/Lighteval-on-LUMI/main/evals/tasks/fineweb.txt"
fi

# Set custom tasks script
custom_tasks_script="./evals/tasks/lighteval_tasks.py"
# Set tasks list file
tasks_list_file="./evals/tasks/fineweb.txt"
# Iterate through the model_folder
echo "Iterating through the model folder."
for model_checkpoint_folder in $model_folder/*; do
    # Check if the model_checkpoint_folder is a directory
    if [ -d "$model_checkpoint_folder" ]; then
        # Check if the model_checkpoint_folder contains a model checkpoint file
        if ! find "$model_checkpoint_folder" -maxdepth 1 -type f \( -name "*.ckpt" -o -name "*.h5" -o -name "*.pth" -o -name "*.pt" -o -name "*.safetensors" \) | grep -q .; then
            echo "Model checkpoint file does not exist in $model_checkpoint_folder."
            continue
        fi
        # Check if the model_checkpoint_folder contains a config file
        if [ ! -f "$model_checkpoint_folder/config.json" ]; then
            echo "Config file does not exist in $model_checkpoint_folder."
            continue
        fi
        # Check if the model_checkpoint_folder contains a tokenizer.json file
        if [ ! -f "$model_checkpoint_folder/tokenizer.json" ]; then
            echo "Tokenizer file does not exist in $model_checkpoint_folder."
            continue
        fi
        # Set the job_name
        job_name="${job_name_prefix}$(basename $model_checkpoint_folder)"
        # Set the job_log
        job_log="$(realpath ${log_output})/${job_name}.o%j.log"
        # Set the job_err
        job_err="$(realpath ${log_output})/${job_name}.e%j.log"
        # Set the job_script
        job_script="$(realpath ${jobs_output})/${job_name}.sh"
        # Set absolute path to eval_output
        job_eval_output=$(realpath $eval_output)/${job_name}
        # Skip if the results folder exists
        if [ -d "$job_eval_output/results" ]; then
            echo "Results already exist for $model_checkpoint_folder. Skipping."
            continue
        fi
        mkdir -p $job_eval_output
        # Set absolute path to model_checkpoint_folder
        model_checkpoint_folder=$(realpath $model_checkpoint_folder)
        # Set absolute path to virtual_env
        virtual_env=$(realpath $virtual_env)
        # Set absolute path to custom_tasks_script
        custom_tasks_script=$(realpath $custom_tasks_script)
        # Set absolute path to tasks_list_file
        tasks_list_file=$(realpath $tasks_list_file)
        # Create the job script
        echo "#!/bin/bash
#SBATCH --job-name=${job_name}
#SBATCH --output=${job_log}
#SBATCH --error=${job_err}
#SBATCH --partition=${partition}
#SBATCH --nodes=1
#SBATCH --gres=gpu:mi250:${gpu}
#SBATCH --time=${time}
#SBATCH --account=${account}

module use /appl/local/csc/modulefiles/
module purge
module load LUMI
module load pytorch

source ${virtual_env}/bin/activate
export PYTHONPATH=${virtual_env}/lib/python3.10/site-packages
export HF_HOME=/scratch/${account}/$USER/hf_cache

srun lighteval accelerate \
    --model_args=\"vllm,pretrained=${model_checkpoint_folder},dtype=${model_dtype}\" \
    --custom_tasks \"${custom_tasks_script}\" --max_samples 1000 \
    --tasks \"${tasks_list_file}\" \
    --output_dir \"${job_eval_output}\"" > "$job_script"
        # Submit the job
        sbatch $job_script
    fi
done
