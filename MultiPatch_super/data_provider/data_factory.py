import sys
# print(sys.path)
sys.path.append("/home/yixin/work/msxf/CQU_TimeSeries_Algo/MultiPatch_super")
from data_provider.data_loader import Dataset_Train_dev, Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_Pred
from torch.utils.data import DataLoader
import argparse


data_dict = {
    'ETTh1': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,
    'ETTm2': Dataset_ETT_minute,
    'custom': Dataset_Custom,
    'filtered_dev': Dataset_Train_dev,
    'train_dev_8': Dataset_Train_dev,
    'train_time': Dataset_Train_dev,
    "one_for_one_pool_1":Dataset_Train_dev,
    "one_for_one_pool_4":Dataset_Train_dev,
    "one_for_one_pool_8":Dataset_Train_dev,
    "one_for_one_pool_9":Dataset_Train_dev,
    "one_for_one_pool_12":Dataset_Train_dev,
    "one_for_one_pool_13":Dataset_Train_dev,
    "one_for_one_pool_14":Dataset_Train_dev,
    "one_for_one_pool_16":Dataset_Train_dev,
    "one_for_one_pool_20":Dataset_Train_dev,
    "one_for_one_pool_23":Dataset_Train_dev,
    "one_for_one_pool_24":Dataset_Train_dev,
    "one_for_one_pool_26":Dataset_Train_dev
}


def data_provider(args, flag):
    Data = data_dict[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1

    if flag == 'test':
        shuffle_flag = False
        drop_last = True
        batch_size = args.batch_size
        freq = args.freq
    elif flag == 'pred':
        shuffle_flag = False
        drop_last = False
        batch_size = 1
        freq = args.freq
        Data = Dataset_Pred
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size # 128
        freq = args.freq # h

    data_set = Data(
        root_path=args.root_path,
        data_path=args.data_path,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features, # M
        target=args.target, # OT
        timeenc=timeenc,
        freq=freq,
        mode=args.mode) # 'one_for_all' or 'one_for_one' in Dataset_Train_dev
    
    print(flag, len(data_set))
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last)
    return data_set, data_loader


def unit_test():
    parser = argparse.ArgumentParser(description='For unit test')
    parser.add_argument('--freq', type=str, default='h')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size of train input data')
    parser.add_argument("--root_path", type=str, default="/home/yixin/work/msxf/CQU_TimeSeries_Algo/MultiPatch_super/dataset/")
    parser.add_argument("--data_path", type=str, default="train_dev1.csv")
    parser.add_argument("--data", type=str, default="train_dev1")
    parser.add_argument('--seq_len', type=int, default=336, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='start token length')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
    parser.add_argument('--features', type=str, default='M')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')

    args = parser.parse_args()
    flag="train"


    data_set, data_loader =  data_provider(args, flag)
    return data_set, data_loader

if __name__ == "__main__":
    data_set, data_loader = unit_test()