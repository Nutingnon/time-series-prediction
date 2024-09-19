# nohup bash ./Autoformer/train_dev1.sh one_for_one_pool_9 336 192 &
# nohup bash ./DLinear/train_dev1.sh one_for_one_pool_9 336 192 &
# nohup bash ./Informer/train_dev1.sh one_for_one_pool_9 336 192 &
# nohup bash ./iTransformer/train_dev1.sh one_for_one_pool_9 336 192 &
# nohup bash ./LSTM/train_dev1.sh one_for_one_pool_9 336 192 &
# nohup bash ./Mamba/train_dev1.sh one_for_one_pool_9 336 192 &
# nohup bash ./MultiPatch/train_dev1.sh one_for_one_pool_9 336 192 &
# nohup bash ./NLinear/train_dev1.sh one_for_one_pool_9 336 192 &
# nohup bash ./PatchTST/train_dev1.sh one_for_one_pool_9 336 192 &
# nohup bash ./TimeMixer/train_dev1.sh one_for_one_pool_9 336 192 &


#!/bin/bash

# Define an array of dataset names
datasets=("one_for_one_pool_12" "one_for_one_pool_13" "one_for_one_pool_14" "one_for_one_pool_16" "one_for_one_pool_20" "one_for_one_pool_23" "one_for_one_pool_24" "one_for_one_pool_26")  # Add more datasets as needed

# Loop over each dataset
for dataset in "${datasets[@]}"
do
    echo "Running with dataset: $dataset"
    nohup bash ./Autoformer/train_dev1.sh "$dataset" 336 192 &
    nohup bash ./DLinear/train_dev1.sh "$dataset" 336 192 &
    nohup bash ./Informer/train_dev1.sh "$dataset" 336 192 &
    nohup bash ./iTransformer/train_dev1.sh "$dataset" 336 192 &
    nohup bash ./LSTM/train_dev1.sh "$dataset" 336 192 &
    nohup bash ./Mamba/train_dev1.sh "$dataset" 336 192 &
    nohup bash ./MultiPatch/train_dev1.sh "$dataset" 336 192 &
    nohup bash ./NLinear/train_dev1.sh "$dataset" 336 192 &
    nohup bash ./PatchTST/train_dev1.sh "$dataset" 336 192 &
    nohup bash ./TimeMixer/train_dev1.sh "$dataset" 336 192 &

    # Optionally, you can add a wait period between iterations if needed
    # sleep 60  # Sleep for 60 seconds between iterations
    wait
    echo "Finished running with dataset: $dataset"
done
