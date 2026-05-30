
import numpy as np
import pandas as pd
import h5py
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset, DataLoader
import warnings
warnings.filterwarnings('ignore')

class NCMAPSSLoaderV2:
    
    def __init__(self, data_dir='./N_MAPSS'):
        self.data_dir = data_dir
        self.variable_names = {
            'X_s': ['T24', 'T30', 'T48', 'T50', 'P15', 'P2', 'P21', 'P24', 
                   'Ps30', 'P40', 'P50', 'Nf', 'Nc', 'Wf'],
            'X_v': ['T40', 'P30', 'P45', 'W21', 'W22', 'W25', 'W31', 'W32',
                   'W48', 'W50', 'SmFan', 'SmLPC', 'SmHPC', 'phi'],
            'T': ['fan_eff_mod', 'fan_flow_mod', 'LPC_eff_mod', 'LPC_flow_mod',
                 'HPC_eff_mod', 'HPC_flow_mod', 'HPT_eff_mod', 'HPT_flow_mod',
                 'LPT_eff_mod', 'LPT_flow_mod'],
            'W': ['alt', 'Mach', 'TRA', 'T2'],
            'A': ['unit', 'cycle', 'Fc', 'hs']
        }
    
    def load_dataset_sampled(self, dataset_name='N-CMAPSS_DS01-005', 
                            max_engines=None, split='dev'):
        file_path = f'{self.data_dir}/{dataset_name}.h5'
        suffix = '_dev' if split == 'dev' else '_test'
        
        with h5py.File(file_path, 'r') as f:
            a_data = f[f'A{suffix}'][:]
            units = np.unique(a_data[:, 0])
            
            if max_engines is not None and len(units) > max_engines:
                units = units[:max_engines]
            
            print(f"加载 {split} 集: {len(units)} 台发动机")
            
            mask = np.isin(a_data[:, 0], units)
            indices = np.where(mask)[0]
            
            data = {}
            chunk_size = 100000
            
            for key in ['A', 'W', 'X_s', 'X_v', 'T', 'Y']:
                dataset_key = f'{key}{suffix}'
                full_data = f[dataset_key]
                
                sampled = full_data[indices]
                data[dataset_key] = sampled
            
            for key in ['A_var', 'W_var', 'X_s_var', 'X_v_var', 'T_var']:
                data[key] = [v.decode() if isinstance(v, bytes) else v for v in f[key][:]]
        
        return data, units
    
    def create_dataframe_sampled(self, data, split='dev'):
        suffix = '_dev' if split == 'dev' else '_test'
        
        df = pd.DataFrame()
        
        a_data = data[f'A{suffix}']
        for i, name in enumerate(data['A_var']):
            df[name] = a_data[:, i]
        
        w_data = data[f'W{suffix}']
        for i, name in enumerate(data['W_var']):
            df[name] = w_data[:, i]
        
        xs_data = data[f'X_s{suffix}']
        for i, name in enumerate(data['X_s_var']):
            df[name] = xs_data[:, i]
        
        xv_data = data[f'X_v{suffix}']
        for i, name in enumerate(data['X_v_var']):
            df[name] = xv_data[:, i]
        
        t_data = data[f'T{suffix}']
        for i, name in enumerate(data['T_var']):
            df[name] = t_data[:, i]
        
        df['RUL'] = data[f'Y{suffix}'].flatten()
        
        return df
    
    def normalize(self, train_df, test_df):
        exclude_cols = ['unit', 'cycle', 'Fc', 'hs', 'RUL']
        feature_cols = [c for c in train_df.columns if c not in exclude_cols]
        
        scaler = StandardScaler()
        train_df[feature_cols] = scaler.fit_transform(train_df[feature_cols])
        test_df[feature_cols] = scaler.transform(test_df[feature_cols])
        
        return train_df, test_df, scaler
    
    def prepare_sequence_data(self, df, seq_length=50, stride=1):
        exclude_cols = ['unit', 'cycle', 'Fc', 'hs', 'RUL']
        feature_cols = [c for c in df.columns if c not in exclude_cols]
        
        sequences = []
        labels = []
        units = []
        
        for unit_id in df['unit'].unique():
            unit_data = df[df['unit'] == unit_id].sort_values('cycle')
            unit_values = unit_data[feature_cols].values
            unit_labels = unit_data['RUL'].values
            
            for i in range(0, len(unit_values) - seq_length + 1, stride):
                seq = unit_values[i:i+seq_length]
                label = unit_labels[i+seq_length-1]
                sequences.append(seq)
                labels.append(label)
                units.append(unit_id)
        
        return np.array(sequences, dtype=np.float32), np.array(labels, dtype=np.float32), np.array(units)

class LazySequenceDataset(Dataset):
    
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return torch.FloatTensor(self.sequences[idx]), torch.FloatTensor([self.labels[idx]])

def create_dataloaders_v2(train_seq, train_labels, test_seq, test_labels, 
                         batch_size=256, val_split=0.1):
    
    n_train = int(len(train_seq) * (1 - val_split))
    
    indices = np.random.permutation(len(train_seq))
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]
    
    train_dataset = LazySequenceDataset(train_seq[train_indices], train_labels[train_indices])
    val_dataset = LazySequenceDataset(train_seq[val_indices], train_labels[val_indices])
    test_dataset = LazySequenceDataset(test_seq, test_labels)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    return train_loader, val_loader, test_loader

def prepare_ncmapss_for_experiment(dataset_name='N-CMAPSS_DS01-005', 
                                   max_engines=6, seq_length=50, stride=5,
                                   include_t_var=True):
    loader = NCMAPSSLoaderV2('./N_MAPSS')
    
    print(f"加载开发集 (最多{max_engines}台发动机)...")
    train_data, train_units = loader.load_dataset_sampled(dataset_name, max_engines, 'dev')
    train_df = loader.create_dataframe_sampled(train_data, 'dev')
    
    print(f"加载测试集 (最多{max_engines}台发动机)...")
    test_data, test_units = loader.load_dataset_sampled(dataset_name, max_engines, 'test')
    test_df = loader.create_dataframe_sampled(test_data, 'test')
    
    print(f"开发集记录数: {len(train_df)}, 测试集记录数: {len(test_df)}")
    
    train_df, test_df, scaler = loader.normalize(train_df, test_df)
    
    if include_t_var:
        feature_cols = [c for c in train_df.columns 
                       if c not in ['unit', 'cycle', 'Fc', 'hs', 'RUL']]
    else:
        t_vars = loader.variable_names['T']
        feature_cols = [c for c in train_df.columns 
                       if c not in ['unit', 'cycle', 'Fc', 'hs', 'RUL'] + t_vars]
    
    print(f"准备序列数据 (stride={stride})...")
    train_seq, train_labels, _ = loader.prepare_sequence_data(
        train_df[['unit', 'cycle', 'Fc', 'hs', 'RUL'] + feature_cols], 
        seq_length=seq_length, stride=stride
    )
    test_seq, test_labels, _ = loader.prepare_sequence_data(
        test_df[['unit', 'cycle', 'Fc', 'hs', 'RUL'] + feature_cols],
        seq_length=seq_length, stride=stride
    )
    
    print(f"训练样本: {len(train_seq)}, 测试样本: {len(test_seq)}")
    print(f"特征维度: {train_seq.shape[-1]}")
    
    train_loader, val_loader, test_loader = create_dataloaders_v2(
        train_seq, train_labels, test_seq, test_labels, batch_size=256
    )
    
    input_dim = train_seq.shape[-1]
    
    return train_loader, val_loader, test_loader, input_dim

if __name__ == '__main__':
    print("=" * 60)
    print("优化版数据加载器测试")
    print("=" * 60)
    
    train_loader, val_loader, test_loader, input_dim = prepare_ncmapss_for_experiment(
        dataset_name='N-CMAPSS_DS01-005',
        max_engines=6,
        seq_length=50,
        stride=5,
        include_t_var=True
    )
    
    for batch_x, batch_y in train_loader:
        print(f"Batch形状: X={batch_x.shape}, Y={batch_y.shape}")
        break
    
    print("\n数据加载器测试完成!")
