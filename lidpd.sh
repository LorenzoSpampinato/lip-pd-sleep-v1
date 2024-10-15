#!/bin/bash
#SBATCH --job-name='lid_pd_1'
#SBATCH -o /home/lorenzo.spampinato/MEDITECH/BSP/projects/lid_pd/outfile_LID_PD_1
#SBATCH -e /home/lorenzo.spampinato/MEDITECH/BSP/projects/lid_pd/errfile_LID_PD_1
#SBATCH -N1
#SBATCH -p gpu
#SBATCH --time 0-00:10:00

# Specify the following variables
ENV_NAME="LID_PD"
VOLUME= "BSP/data/"
PWD_APP="/home/lorenzo.spampinato/scratch/0_lid-pd/"#working directory
IMAGE_PATH="/home/lorenzo.spampinato/MEDITECH/BSP/singularity_img/miniconda3_lidpd.sif" #dove si trova singularity image

DATA_PATH= "D://TESI//lid-data-samples//lid-data-samples//Dataset"
LABEL_PATH="D://TESI//lid-data-samples//lid-data-samples//Labels"
SAVE_PATH="D://TESI//lid-data-samples//lid-data-samples//Results_prova"
INFO_PATH="D://TESI//lid-data-samples//lid-data-samples//Results_prova"

#DATA_PATH="/app/data/lid_pd/Dataset"
#LABEL_PATH="/app/data/lid_pd/Labels"
#SAVE_PATH="/app/data/lid_pd/Results"
#INFO_PATH="/app/data/lid_pd/Results"

RUN_FEATURES=true
ONLY_CLASS="DYS"
ONLY_PATIENT="PD012"

# Specify the python scripts to run
PYTHON_SCRIPT="C:\Users\Lorenzo\PycharmProjects\lid-pd-sleep-v0"
PYTHON_ARGS="--data_path $DATA_PATH --label_path $LABEL_PATH --save_path $SAVE_PATH --info_path $INFO_PATH --run_features $RUN_FEATURES --only_class $ONLY_CLASS --only_patient $ONLY_PATIENT"

srun singularity exec -B $VOLUME=/data/, $VOLUME= /output/ $PWD_APP:/app/ --pwd /app $IMAGE_PATH \
bash -c "source /opt/conda/etc/profile.d/conda.sh && \
conda activate $ENV_NAME && \
python $PYTHON_SCRIPT $PYTHON_ARGS"
