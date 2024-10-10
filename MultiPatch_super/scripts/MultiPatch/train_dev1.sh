if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi
seq_len=336
model_name=MultiPatch

root_path_name=../../dataset/
# data_path_name=train_dev1.csv
# model_id_name=train_dev1
# data_name=train_dev1
data_path_name=filtered_dev.csv
model_id_name=filtered_dev
data_name=filtered_dev

random_seed=2021
for pred_len in 192
do
    python -u ../../run_longExp.py \
        --random_seed $random_seed \
        --is_training 1 \
        --root_path $root_path_name \
        --data_path $data_path_name \
        --model_id $model_id_name_$seq_len'_'$pred_len \
        --model $model_name \
        --data $data_name \
        --features M \
        --seq_len $seq_len \
        --pred_len $pred_len \
        --enc_in 4 \
        --e_layers 3 \
        --n_heads 8 \
        --d_model 128 \
        --individual 1\
        --d_ff 256 \
        --dropout 0.2\
        --fc_dropout 0.2\
        --head_dropout 0\
        --patch_len '3,9,12'\
        --stride 5\
        --des 'Exp' \
        --train_epochs 50\
        --itr 5 \
        --batch_size 512 \
        --learning_rate 0.001 > logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_bz512_patchlen3912_epoch50_individual_huberloss_lr0.001_'$pred_len.log
done