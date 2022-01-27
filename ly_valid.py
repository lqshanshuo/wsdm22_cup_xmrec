import sys

sys.path.insert(1, 'src')
from model import Model
from utils import *
from data import *
from ly_train import *
import pdb

def run_valid(args):

    if torch.cuda.is_available() and args.cuda:
        torch.cuda.set_device(0)
    args.device = torch.device('cuda' if torch.cuda.is_available() and args.cuda else 'cpu')
    print("Device:", args.device)

    # load pretrained model
    model_dir = f'checkpoints/{args.tgt_market}_{args.src_markets}_{args.exp_name}.model'
    id_bank_dir = f'checkpoints/{args.tgt_market}_{args.src_markets}_{args.exp_name}.pickle'

    with open(id_bank_dir, 'rb') as centralid_file:
        my_id_bank = pickle.load(centralid_file)

    mymodel = Model(args, my_id_bank)
    mymodel.load(model_dir)

    ############
    ## Validation Run
    ############

    print('Loaded target data!\n')
    train_file_names = args.train_data_file # 'train_5core.tsv', 'train.tsv' for the original data loading
    tgt_train_data_dir = os.path.join(args.data_dir, args.tgt_market, train_file_names)
    tgt_train_ratings = pd.read_csv(tgt_train_data_dir, sep='\t')
    tgt_task_generator = TaskGenerator(tgt_train_ratings, my_id_bank)
    tgt_valid_dataloader = tgt_task_generator.instance_a_market_valid_dataloader(f'DATA/{args.tgt_market}/valid_run.tsv', args.batch_size)

    print('Validation output:')
    # validation data prediction
    valid_run_mf = mymodel.predict(tgt_valid_dataloader)
    tgt_valid_qrel = read_qrel_file(f'DATA/{args.tgt_market}/valid_qrel.tsv')
    task_ov, task_ind = get_evaluations_final(valid_run_mf, tgt_valid_qrel)
    print(task_ov)

    # valid_output_file = f'valid_{args.tgt_market}_{args.src_markets}_{args.exp_name}.tsv'
    # print(f'--validation: {valid_output_file}')
    # write_run_file(valid_run_mf, valid_output_file)

def main():
    parser = create_arg_parser()
    args = parser.parse_args()
    args.batch_size = 5000
    run_valid(args)

print('Validation finished successfully!')

if __name__=="__main__":
    main()
