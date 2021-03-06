#!/bin/bash

ENV="HandReach20-v0"
#NUM_TIMESTEPS="10000"
#N_EPOCHS="301"
SAVE_VIDEO_INTERVAL="100000000"
BASE_LOG_PATH="logs"
MODE="maximum_span"
MCA_EXPXLORATION="eps_greedy"
MCA_ACTION_L2="0"
SHARING="False"
NUM_ENV="16"
MAX_U="1"
SS="False"
DILUTE_AT_GOAL="False"
COVER_DISTANCE_TH=0.08

k=0
kmax=110
delta_k=20

DATE=`date +%Y-%m-%d`

while [ $k -le $kmax ]
do
    k=$(( $k + $delta_k ))

    #######################
    #### Learn k-Random cover
    RANDOM_COVER="True"
    N_EPOCHS="50"
    LOG_PATH="${BASE_LOG_PATH}/${DATE}-${ENV}/K${k}/random"

    echo "          -------------------------------------------------------------------------------------
          -------------------------------------------------------------------------------------
          -------------------------------------------------------------------------------------
          Starting new experiment. Logging directory ${LOG_PATH}
          -------------------------------------------------------------------------------------
          -------------------------------------------------------------------------------------
          -------------------------------------------------------------------------------------"

     xvfb-run -a -s "-screen 0 1400x900x24" /home/ron/nir/work/git/venv/gym_venv_mj150/bin/python3.6 -m baselines.run --alg=her --env=$ENV\
     --n_epochs=$N_EPOCHS --save_video_interval=$SAVE_VIDEO_INTERVAL --log_path=$LOG_PATH --mode=$MODE --num_env=$NUM_ENV\
     --max_u=$MAX_U --ss=$SS --k=$k --random_cover=$RANDOM_COVER --dilute_at_goal=$DILUTE_AT_GOAL\
     --trainable='False'

    #######################
    #### Learn k-MC cover
    RANDOM_COVER="False"
    N_EPOCHS="800"
    LOG_PATH="${BASE_LOG_PATH}/${DATE}-${ENV}/K${k}/learned"
    echo "          -------------------------------------------------------------------------------------
          -------------------------------------------------------------------------------------
          -------------------------------------------------------------------------------------
          Starting new experiment. Logging directory ${LOG_PATH}
          -------------------------------------------------------------------------------------
          -------------------------------------------------------------------------------------
          -------------------------------------------------------------------------------------"

    xvfb-run -a -s "-screen 0 1400x900x24" /home/ron/nir/work/git/venv/gym_venv_mj150/bin/python3.6 -m baselines.run --alg=her --env=$ENV\
     --n_epochs=$N_EPOCHS --save_video_interval=$SAVE_VIDEO_INTERVAL --log_path=$LOG_PATH --mode=$MODE --num_env=$NUM_ENV\
      --max_u=$MAX_U --ss=$SS --k=$k --random_cover=$RANDOM_COVER --dilute_at_goal=$DILUTE_AT_GOAL\

done
