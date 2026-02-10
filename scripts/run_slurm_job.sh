#!/bin/bash
#SBATCH --job-name=llm_txt_gen
#SBATCH --chdir=/net/tscratch/people/{your plgrid login}/llm_txt_gen/project
#SBATCH --output=logs/llm_txt_gen/vllm_job_%A_task_%a.out
#SBATCH --error=logs/llm_txt_gen/vllm_job_%A_task_%a.err
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --account=your_gpu_grant_name
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00



# Load slurm config file
CONFIG_FILE_PATH="slurm_job.conf"
if [ -f "$CONFIG_FILE_PATH" ]; then
    echo "Loading configuration from $CONFIG_FILE_PATH"
    set -a 
    source "$CONFIG_FILE_PATH"
    set +a 
else
    echo "ERROR: Configuration file '$CONFIG_FILE_PATH' not found."
    exit 1
fi


# Project root directory
PROJECT_DIR="${PROJECT_ROOT_ON_SCRATCH}/${PROJECT_SUBDIR_NAME}"
cd "$PROJECT_DIR" || { echo "ERROR: Failed to change directory into ${PROJECT_DIR}"; exit 1; }


# Slurm logs directory
mkdir -p "${SLURM_LOG_DIR_RELATIVE_TO_PROJECT}"


echo "Job ID: $SLURM_JOB_ID"
echo "Job Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Node list: $SLURM_JOB_NODELIST"
echo "Number of nodes: $SLURM_JOB_NUM_NODES"
echo "GPUs per node: $SLURM_GPUS_ON_NODE"
echo "Tasks per node: $SLURM_NTASKS_PER_NODE"
echo "CPUs per task: $SLURM_CPUS_PER_TASK"
echo "Memory per task: $SLURM_MEM_PER_NODE"
echo "Working directory: $(pwd)"


# Activate Conda environment
module load "${MINICONDA_MODULE}"
eval "$(conda shell.bash hook)"
CONDA_ENV_PATH="${SCRATCH}/.conda/envs/${CONDA_ENV_NAME}"
if [ -d "$CONDA_ENV_PATH" ]; then
    conda activate "$CONDA_ENV_PATH"
    echo "Conda environment '${CONDA_ENV_NAME}' activated"
else
    echo "ERROR: Conda environment '${CONDA_ENV_NAME}' not found at $CONDA_ENV_PATH"
    exit 1
fi


# Set Hugging Face cache to the SCRATCH
export HF_HOME="${SCRATCH}/${HF_CACHE_SUBDIR}"
mkdir -p "$HF_HOME"
echo "HF_HOME set to $HF_HOME"


if [ -f "${HF_HOME}/token" ]; then
    chmod 600 "${HF_HOME}/token"
else
    echo "WARNING: Hugging Face token file not found at ${HF_HOME}/token."
fi


# Set VLLM cache directory to the SCRATCH
export VLLM_CACHE_DIR="${SCRATCH}/${VLLM_CACHE_SUBDIR}"
mkdir -p "$VLLM_CACHE_DIR"
echo "VLLM_CACHE_DIR set to $VLLM_CACHE_DIR"


# Set torchinductor cache to the SCRATCH
export TORCHINDUCTOR_CACHE_DIR="${SCRATCH}/${TORCHINDUCTOR_CACHE_SUBDIR}"
mkdir -p "$TORCHINDUCTOR_CACHE_DIR"
echo "TORCHINDUCTOR_CACHE_DIR set to $TORCHINDUCTOR_CACHE_DIR"


# Log GPU info
if command -v nvidia-smi &> /dev/null
then
    echo "--- nvidia-smi output ---"
    nvidia-smi
    echo "-------------------------"
else
    echo "nvidia-smi not found"
fi


# Start main.py script
echo "Starting Python main.py script..."
python main.py
echo "Python main.py script finished."

conda deactivate
echo "Job finished."