ce_loss_weight=0.1
tmp=0.1
ang_weight=0.1
stage_two_lr=1e-4
seed=1
dataset=$1

model_path=$2
dir_name=$(basename "${model_path}")
mkdir -p "./emo_anchors/${dir_name}"

python src/generate_anchors.py --bert_path $model_path

CUDA_VISIBLE_DEVICES=3 python src/run.py --anchor_path "./emo_anchors/${dir_name}" \
                                         --bert_path $model_path \
                                         --dataset_name $dataset \
                                         --ce_loss_weight $ce_loss_weight \
                                         --temp $tmp \
                                         --seed $seed \
                                         --angle_loss_weight $ang_weight \
                                         --stage_two_lr $stage_two_lr \
                                         --disable_training_progress_bar \
                                         --use_nearest_neighbour > output.log &
