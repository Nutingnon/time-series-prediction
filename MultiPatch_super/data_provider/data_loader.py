import os
import numpy as np
import pandas as pd
import os
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features
import warnings
from collections import Counter
from typing import List

warnings.filterwarnings('ignore')


class Dataset_Train_dev(Dataset):
    """
    A custom dataset class for loading and processing time-series data for training, validation, and testing.
    
    Attributes:
        root_path (str): The root directory containing the dataset.
        flag (str): A flag to indicate the mode of the dataset. It can be 'train', 'val', or 'test'.
        size (list): A list that defines the input sequence length, label length, and prediction length.
        features (str): Indicates the type of features to use ('S' for single target, 'M' for multi-variable, 'MS' for multi-step).
        data_path (str): The filename of the dataset to load.
        target (str): The name of the target variable for supervised learning.
        scale (bool): Whether to apply scaling to the data.
        timeenc (int): Time encoding method (0 for categorical encoding, 1 for continuous time encoding).
        freq (str): The frequency of the time series data, e.g., 'h' for hourly.
    """
    def __init__(self, root_path, flag='train', size=None,
                    features='S', data_path='train_dev_8.csv',
                    target='cpu_usage', scale=True, timeenc=0, freq='h', mode='one_for_all'):
        """
        Initializes the dataset by setting parameters and loading data from the specified path.
        
        Args:
            root_path (str): Directory of the dataset.
            flag (str): Mode of operation ('train', 'test', 'val').
            size (list or None): Sequence size information in the form [seq_len, label_len, pred_len].
            features (str): Type of features to use ('S', 'M', 'MS').
            data_path (str): The filename of the data file to load.
            target (str): The target variable name.
            scale (bool): Whether to scale the input data.
            timeenc (int): Time encoding strategy (0 for categorical, 1 for continuous).
            freq (str): Frequency of the data (e.g., 'h' for hourly).
        """

        # size [seq_len, label_len, pred_len]
        # Set default sequence sizes if none provided
        if size is None:
            self.seq_len = 24 * 4 * 4  # 384 time steps (typically hourly data over 16 days)
            self.label_len = 24 * 4    # 96 time steps (typically 4 days)
            self.pred_len = 24 * 4     # 96 time steps (prediction horizon)

        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
            
        print("size::::", size[0], size[1], size[2])
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        self.flag = flag

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.mode = mode
        self.__read_data__()

    def __read_data__(self):
        """
        Loads the data, scales it, and processes the timestamp information.
        """
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))
        self.value_columns = df_raw.columns[2:]
        if self.mode == 'one_for_all':
            pool_ids = df_raw['pool_id'].unique()
            test_pool_id: List = np.random.choice(pool_ids, 3, replace=False).tolist()
            valid_pool_id = test_pool_id.pop()
            df_train = df_raw.loc[~df_raw['pool_id'].isin(test_pool_id),:]
            df_test = df_raw.loc[df_raw['pool_id'].isin(test_pool_id), :]
            df_valid = df_raw.loc[df_raw['pool_id']==valid_pool_id, :]
            self.df_train = df_train.reset_index(drop=True)
            self.df_test = df_test.reset_index(drop=True)
            self.df_valid = df_valid.reset_index(drop=True)

            num_train = self.df_train.shape[0]
            num_test = self.df_test.shape[0]
            num_vali = self.df_valid.shape[0]
            border1s = [0, num_train - self.seq_len, num_train + num_vali - self.seq_len]
            border2s = [num_train, num_train + num_vali, num_train+num_vali+num_test]
            border1 = border1s[self.set_type]
            border2 = border2s[self.set_type]
            columns = self.df_train.columns
            # value_columns = columns[2:]

            if self.features == 'M' or self.features == 'MS':
                # cols_data = df_raw.columns[2:]  # Exclude timestamp 
                # df_data = df_raw[cols_data]
                pass
            elif self.features == 'S':
                self.df_train = self.df_train.loc[:, columns[:2] + [self.target]]
                self.df_test = self.df_test.loc[:, columns[:2] + [self.target]]
                self.df_valid = self.df_valid.loc[:, columns[:2] + [self.target]]

            if self.scale:
                self.df_train.loc[:, self.value_columns] = self.scaler.fit_transform(self.df_train.loc[:, self.value_columns])
                self.df_test.loc[:, self.value_columns] = self.scaler.fit_transform(self.df_test.loc[:, self.value_columns])
                self.df_valid.loc[:, self.value_columns] = self.scaler.fit_transform(self.df_valid.loc[:, self.value_columns])

            # df_stamp = df_raw[['timestamp']][border1:border2]
            df_stamp = dict()
            data_stamp = dict()
            df_stamp['train'] = self.df_train[['timestamp']]
            df_stamp['val'] = self.df_valid[['timestamp']]
            df_stamp['test'] = self.df_test[['timestamp']]
            for key in df_stamp.keys():
                df_stamp[key]['timestamp'] = pd.to_datetime(df_stamp[key]['timestamp'], unit='s')
            if self.timeenc == 0:
                for key in df_stamp.keys():
                    df_stamp[key]['month'] = df_stamp[key].timestamp.apply(lambda row: row.month)
                    df_stamp[key]['day'] = df_stamp[key].timestamp.apply(lambda row: row.day)
                    df_stamp[key]['weekday'] = df_stamp[key].timestamp.apply(lambda row: row.weekday())
                    df_stamp[key]['hour'] = df_stamp[key].timestamp.apply(lambda row: row.hour)
                    data_stamp[key] = df_stamp[key].drop(['timestamp'], axis=1).values
            elif self.timeenc == 1:
                for key in df_stamp.keys():
                    data_stamp[key] = time_features(pd.to_datetime(df_stamp[key]['timestamp'].values), freq=self.freq)
                    data_stamp[key] = data_stamp[key].transpose(1, 0)

            self.data_stamp_all = data_stamp
            self.data_stamp = self.data_stamp_all[self.flag]
            if self.set_type == 0:
                self.data_x = self.df_train
                self.data_y = self.df_train
            elif self.set_type == 1:
                self.data_x = self.df_valid
                self.data_y = self.df_valid
                
            elif self.set_type == 2:
                self.data_x = self.df_test
                self.data_y = self.df_test
            self.data_x.reset_index(inplace=True,drop=True)
            self.data_y.reset_index(inplace=True,drop=True)
            print("self.data_x:",self.data_x.head())
            print("self.data_stamp:",self.data_stamp[:5])



        # one for one
        elif self.mode=="one_for_one":
            num_train = int(len(df_raw) * 0.7)
            num_test = int(len(df_raw) * 0.2)
            num_vali = len(df_raw) - num_train - num_test
            border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len] 
            border2s = [num_train, num_train + num_vali, len(df_raw)] 
            border1 = border1s[self.set_type] # train = 0 -- 0 | start ?
            border2 = border2s[self.set_type] # train = 0 -- num_train | end ?

            if self.features == 'M' or self.features == 'MS':
                # value_columns = df_raw.columns[2:]  # Exclude timestamp 
                df_data = df_raw.reset_index(drop=True)
                
            elif self.features == 'S':
                df_data = df_raw.loc[:, [df_raw.columns[0], df_raw.columns[1], self.target]].reset_index(drop=True)

            if self.scale:
                df_data.loc[:, self.value_columns] = self.scaler.fit_transform(df_data.loc[:, self.value_columns])
                # df_data = self.scaler.transform(df_data.values)
            else:
                pass


            df_stamp = df_raw[['timestamp']][border1: border2].reset_index(drop=True)
            print("df_stamp shape before change:", df_stamp.shape)
            df_stamp['timestamp'] = pd.to_datetime(df_stamp.timestamp, unit='s')
            if self.timeenc == 0:
                df_stamp['month'] = df_stamp.timestamp.apply(lambda row: row.month)
                df_stamp['day'] = df_stamp.timestamp.apply(lambda row: row.day)
                df_stamp['weekday'] = df_stamp.timestamp.apply(lambda row: row.weekday())
                df_stamp['hour'] = df_stamp.timestamp.apply(lambda row: row.hour)
                data_stamp = df_stamp.drop(['timestamp'], axis=1).values
            elif self.timeenc == 1:
                data_stamp = time_features(pd.to_datetime(df_stamp['timestamp'].values), freq=self.freq)
                data_stamp = data_stamp.transpose(1, 0)

            self.data_x = df_data.iloc[border1:border2, :].reset_index(drop=True)
            self.data_y = df_data.iloc[border1:border2, :].reset_index(drop=True) # here self.data_x == self.data_y
            self.data_stamp = data_stamp

            # print("self.data_x:",self.data_x.shape)
            # print("self.data_stamp:",self.data_stamp.shape)


    def __getitem__(self, index):
        """
        Returns a data sample for a given index, including the input sequence, target sequence, and time encoding.

        Args:
            index (int): The index of the data sample.

        Returns:
            tuple: A tuple containing:
                - seq_x: Input sequence for the model.
                - seq_y: Target sequence for prediction.
                - seq_x_mark: Time encoding for the input sequence.
                - seq_y_mark: Time encoding for the target sequence.
        """

        # s_begin = index
        # s_end = s_begin + self.seq_len  # 336
        # r_begin = s_end - self.label_len # label_len 48
        # r_end = r_begin + self.label_len + self.pred_len # label_len:48 + pred_len: x

        # seq_x = self.data_x[s_begin:s_end]
        # seq_y = self.data_y[r_begin:r_end]
        # seq_x_mark = self.data_stamp[s_begin:s_end]
        # seq_y_mark = self.data_stamp[r_begin:r_end]

        # yixin change start
        s_begin = index
        s_end = s_begin + self.seq_len  # 336
        r_begin = s_end - self.label_len # label_len 48
        r_end = r_begin + self.label_len + self.pred_len # label_len:48 + pred_len: x

        if self.data_x.iloc[s_begin:r_end, 0].nunique() >= 2:
            ids_selected = self.data_x.iloc[s_begin:r_end, [0]] # pool_ids 
            ids_cnt = ids_selected.value_counts()
            remove_id = ids_selected.iloc[-1, 0]
            keep_id = ids_selected.iloc[0,0]
            move_length = ids_cnt[remove_id]

            s_begin = index - move_length
            s_end = s_begin + self.seq_len
            r_begin = s_end - self.label_len
            r_end = r_begin + self.label_len + self.pred_len

            error_ = "before move:\n"
            error_ += f"chosen pool_id: {keep_id} \n"
            s_begin = index
            s_end = s_begin + self.seq_len  # 336
            r_begin = s_end - self.label_len # label_len 48
            r_end = r_begin + self.label_len + self.pred_len # label_len:48 + pred_len: x
            error_ += f"data_y: {self.data_y.iloc[r_begin:r_end, 0].value_counts()} \n"
            error_ += f"data_x: {self.data_x.iloc[s_begin:s_end, 0].value_counts()}, \n"
            error_ += "After move\n"
            s_begin = index - move_length
            s_end = s_begin + self.seq_len
            r_begin = s_end - self.label_len
            r_end = r_begin + self.label_len + self.pred_len
            error_ += f"index: {index}, \n"
            error_ += f"data_y: {self.data_y.iloc[r_begin:r_end, 0].value_counts()} \n"
            error_ += f"data_x: {self.data_x.iloc[s_begin:s_end, 0].value_counts()}, \n"
            error_ += f"move_length:{move_length}"
            assert self.data_x.iloc[s_begin:r_end, 0].nunique() < 2, error_


        seq_x = self.data_x.iloc[s_begin:s_end, 2:].values
        seq_y = self.data_y.iloc[r_begin:r_end, 2:].values
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]         
        # yixin change end
        # if self.flag=='test':
        #     print(len(seq_x),'-',len(seq_y),'-', len(seq_x_mark),'-', len(seq_y_mark),'=========')
        # print(len(seq_x),'-',len(seq_y),'-', len(seq_x_mark),'-', len(seq_y_mark),'=========')


        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_ETT_hour(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h'):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        print("size::::", size[0], size[1], size[2])
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
        border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_ETT_minute(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTm1.csv',
                 target='OT', scale=True, timeenc=0, freq='t'):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12 * 30 * 24 * 4 - self.seq_len, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.seq_len]
        border2s = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Custom(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h'):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]
        # print(cols)
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            # print(self.scaler.mean_)
            # exit()
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
    

class Dataset_Pred(Dataset):
    def __init__(self, root_path, flag='pred', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, inverse=False, timeenc=0, freq='15min', cols=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['pred']

        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        self.cols = cols
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        if self.cols:
            cols = self.cols.copy()
            cols.remove(self.target)
        else:
            cols = list(df_raw.columns)
            cols.remove(self.target)
            cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]
        border1 = len(df_raw) - self.seq_len
        border2 = len(df_raw)

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            self.scaler.fit(df_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        tmp_stamp = df_raw[['date']][border1:border2]
        tmp_stamp['date'] = pd.to_datetime(tmp_stamp.date)
        pred_dates = pd.date_range(tmp_stamp.date.values[-1], periods=self.pred_len + 1, freq=self.freq)

        df_stamp = pd.DataFrame(columns=['date'])
        df_stamp.date = list(tmp_stamp.date.values) + list(pred_dates[1:])
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        if self.inverse:
            self.data_y = df_data.values[border1:border2]
        else:
            self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        if self.inverse:
            seq_y = self.data_x[r_begin:r_begin + self.label_len]
        else:
            seq_y = self.data_y[r_begin:r_begin + self.label_len]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


