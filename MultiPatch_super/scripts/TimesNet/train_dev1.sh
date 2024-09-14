if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi
seq_len=336
model_name=TimesNet

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
      --n_heads 4 \
      --d_model 16 \
      --d_ff 128 \
      --dropout 0.3\
      --fc_dropout 0.3\
      --head_dropout 0\
      --patch_len '357'\
      --stride 5\
      --des 'Exp' \
      --train_epochs 50\
      --itr 5 \
      --batch_size 128 \
      --top_k 5 \
      --num_kernels 6\
      --learning_rate 0.001 >logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_bz128_epoch50_lr0.001_'$pred_len.log
done