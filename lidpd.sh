#!/bin/bash
#SBATCH --job-name='lid_pd_1'
#SBATCH -o /home/gianpaolo.palo/MEDITECH/BSP/shared_projects/lid_pd/outfile_LID_PD_1
#SBATCH -e /home/gianpaolo.palo/MEDITECH/BSP/shared_projects/lid_pd/errfile_LID_PD_1
#SBATCH -N1
#SBATCH -p compute
#SBATCH --time 10-00:00:00

# Specify the following variables
ENV_NAME="LID_PD"
PWD_APP="/home/gianpaolo.palo/MEDITECH/BSP"
IMAGE_PATH="/home/gianpaolo.palo/MEDITECH/BSP/singularity_img/miniconda3_lidpd.sif"

DATA_PATH= "D://TESI//lid-data-samples//lid-data-samples//Dataset"
LABEL_PATH="D://TESI//lid-data-samples//lid-data-samples//Labels"
SAVE_PATH="D://TESI//lid-data-samples//lid-data-samples//Results_prova"
INFO_PATH="D://TESI//lid-data-samples//lid-data-samples//Results_prova"

RUN_FEATURES=true
ONLY_CLASS="DYS"
ONLY_PATIENT="PD012"

# Specify the python scripts to run
PYTHON_SCRIPT="C:\Users\Lorenzo\PycharmProjects\lid-pd-sleep-v0"
PYTHON_ARGS="--data_path $DATA_PATH --label_path $LABEL_PATH --save_path $SAVE_PATH --info_path $INFO_PATH --run_features $RUN_FEATURES --only_class $ONLY_CLASS --only_patient $ONLY_PATIENT"

srun singularity exec -B $PWD_APP:/app/ --pwd /app $IMAGE_PATH \
bash -c "source /opt/conda/etc/profile.d/conda.sh && \
conda activate $ENV_NAME && \
python $PYTHON_SCRIPT $PYTHON_ARGS"
