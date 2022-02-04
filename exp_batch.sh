$cat exp_batch.sh
#!/bin/bash
# set -x
tasks=('s1 s2-s3' 's2 s1-s3' 's3 s1-s2')
models=(dnn sb cs mmoe cgc adi)
epoch=5
exp_name=$1

len_task=${#tasks[@]}
len_model=${#models[@]}
mkdir -p logs

for i in "${tasks[@]}" ; do
{
  for j in "${models[@]}" ; do
  {
    sleep 1
    task=($i)
    time=$(date "+%Y%m%d%H%M%S")
    log=logs/train_${task[0]}_${task[1]}_${exp_name}_${j}.out.${time}
    python ly_train.py --tgt_market ${task[0]} --src_markets ${task[1]} --exp_name ${exp_name}_$j --num_epoch $epoch --cuda --hidden_units 64,32,16,8  --num_negative 9 --model_name $j 2>&1 | tee ${log}
  }&
  done
}
done

for i in $(seq 1 $len_task) ; do
{
  for j in $(seq 1 $len_model) ; do
  {
    job_idx=$[($i-1)*$len_model+$j]
    echo $job_idx
    wait %$job_idx
    task=(${tasks[i-1]})
    time=$(date "+%Y%m%d%H%M%S")
    log=logs/valid_${task[0]}_${task[1]}_${exp_name}_${models[j-1]}.out.${time}
    python ly_valid.py --tgt_market ${task[0]} --src_markets ${task[1]} --exp_name ${exp_name}_${models[j-1]} --cuda --hidden_units 64,32,16,8 --model_name ${models[j-1]} 2>&1 | tee ${log}
  }
  done
}
done
