#!/bin/bash
#SBATCH --job-name='lid_pd_edf'
#SBATCH -o /home/lorenzo.spampinato/scratch/0_lid-pd/outfile_lid_pd_4
#SBATCH -e /home/lorenzo.spampinato/scratch/0_lid-pd/errfile_lid_pd_4
#SBATCH -N1
#SBATCH -p compute

#SBATCH --time 3-00:00:00

# Specify the following variables
ENV_NAME="LID_PD"
VOLUME="/home/lorenzo.spampinato/MEDITECH/BSP/data/lid_pd/Dataset"
#VOLUME="/home/lorenzo.spampinato/MEDITECH/BSP/data/lid_pd/Results/Preprocessed_BIN"
PWD_APP="/home/lorenzo.spampinato/scratch/0_lid-pd/"
IMAGE_PATH="/home/lorenzo.spampinato/MEDITECH/BSP/singularity_img/miniconda3_lidpd.sif"

DATA_PATH="/home/lorenzo.spampinato/MEDITECH/BSP/data/lid_pd/Dataset"
#DATA_PATH="/home/lorenzo.spampinato/MEDITECH/BSP/data/lid_pd/Results/Preprocessed_BIN"
LABEL_PATH="/home/lorenzo.spampinato/MEDITECH/BSP/data/lid_pd/Labels"
SAVE_PATH="/home/lorenzo.spampinato/MEDITECH/BSP/data/lid_pd/Results"
INFO_PATH="/home/lorenzo.spampinato/MEDITECH/BSP/data/lid_pd/Results"


ONLY_CLASS="DYS"
ONLY_PATIENT="PD011"

# Specify the python scripts to run
#PYTHON_SCRIPT="C:\Users\Lorenzo\PycharmProjects\lid-pd-sleep-v0"
PYTHON_SCRIPT="utilities\save_edf.py"
PYTHON_ARGS="--data_path $DATA_PATH --label_path $LABEL_PATH --save_path $SAVE_PATH --info_path $INFO_PATH --only_class ${ONLY_CLASS} --only_patient ${ONLY_PATIENT}"

#srun singularity exec -B $VOLUME:/Preprocessed_BIN/,$PWD_APP:/app/ --pwd /app $IMAGE_PATH \
#srun singularity exec -B ${VOLUME}:/Dataset, ${PWD_APP}:/app --pwd /app ${IMAGE_PATH}
#bash -c "source /opt/conda/etc/profile.d/conda.sh && \
#conda activate $ENV_NAME && \
#python $PYTHON_SCRIPT $PYTHON_ARGS"
srun singularity exec \
    -B ${VOLUME}:/Dataset,${PWD_APP}:/app,${SAVE_PATH}:/Results,${LABEL_PATH}:/Labels \
    --pwd /app ${IMAGE_PATH} \
    bash -c "source /opt/conda/etc/profile.d/conda.sh && \
    conda activate ${ENV_NAME} && \
    python /app/${PYTHON_SCRIPT} --data_path /Dataset --label_path /Labels --save_path /Results --info_path /Results --only_class ${ONLY_CLASS} --only_patient ${ONLY_PATIENT}"