#!/bin/bash
#SBATCH --job-name='lid_pd_1'
#SBATCH -o /home/lorenzo.spampinato/scratch/lid-pd_0/outfile_LID_PD_analysis
#SBATCH -e /home/lorenzo.spampinato/scratch/lid-pd_0/errfile_LID_PD_analysis
#SBATCH -N1
#SBATCH -p compute
#SBATCH --time 10-00:00:00

ENV_NAME="LID_PD"
PWD_APP="/home/lorenzo.spampinato/scratch"
IMAGE_PATH="/home/lorenzo.spampinato/MEDITECH/BSP/singularity_img/miniconda3_lidpd.sif"

DATA_PATH="/app/lid-data-samples/Dataset" #ma sono in scratch, non in data!!!!!!
SAVE_PATH="/app/lid-data-samples/preprocessed"
INFO_PATH="/app/lid-pd_0/results"
LABEL_PATH="/app/lid-data-samples/Labels"

#DEVO RENDERE RELATIVI I PERCORSI SOPRA?


ONLY_CLASS="ADV"
ONLY_PATIENT="PD002"

PYTHON_SCRIPT="/app/lid-pd_0/lidpd_main.py"

PYTHON_ARGS="--data_path $DATA_PATH \
             --save_path $SAVE_PATH \
             --info_path $INFO_PATH \
             --label_path $LABEL_PATH \
             --only_class $ONLY_CLASS \
             --only_patient $ONLY_PATIENT \

srun singularity exec -B $PWD_APP:/app/ --pwd /app $IMAGE_PATH \
bash -c "source /opt/conda/etc/profile.d/conda.sh && \
conda activate $ENV_NAME && \
python $PYTHON_SCRIPT $PYTHON_ARGS"