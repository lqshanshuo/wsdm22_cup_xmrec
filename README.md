# WSDM 2022 CUP - Cross-Market Recommendation - Starter Kit
This repository provides code for training (GMF\DNN\SharedBottom\CrossStitch\MMoE\CGC\ADI) model over several markets. We provide loading data from zero to a few source markets to augment the target market data, which can help the recommendation performance in the target market.

For further information please refer to the competition website: [WSDM_2022_CUP](https://xmrec.github.io/wsdmcup/)

## Requirements:
We use conda for our experimentations. You can use `environment.yml` to create the environment (use `conda env create -f environment.yml`) or install the below list of requirements on your own environment.

- python 3.7
- pandas & numpy (pandas-1.3.3, numpy-1.21.2)
- torch==1.9.1
- [pytrec_eval](https://github.com/cvangysel/pytrec_eval)




## Train model:
`train.py` is the script for training simple GMF++ model (or other DNN SharedBottom CrossStitch MMoE CGC ADI) that is taking one target market and zero to a few source markets for augmenting with the target market. We implemented our dataloader such that it loads all the data and samples equally from each market in the training phase. You can use ConcatDataset from `torch.utils.data` to concatenate your torch Datasets.


Here is a sample train script using two source markets:

    python train.py --tgt_market s1 --src_markets s2-s3 --exp_name my_exp --num_epoch 5 --cuda --model_name adi

Here is a sample test script using target market validation data:

    python valid.py --tgt_market s1 --src_markets s2-s3 --exp_name my_exp --cuda --model_name adi

Here is a sample script to start training tasks in batch:

    sh exp_batch.sh my_exp_batch

When the `exp_batch.sh` task execute is finished, the validation metrics will be recorded in logs/valid_s1_s2-s3_my_exp_batch_adi.out.20220204140856

