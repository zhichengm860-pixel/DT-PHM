
import numpy as np
import pandas as pd
import h5py
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
import warnings
warnings.filterwarnings('ignore')

class CMAPSSLoader:
    
    def __init__(self, data_dir='./C_MAPSS'):
        self.data_dir = data_dir
        self.column_names = ['unit', 'cycle', 'setting_1', 'setting_2', 'setting_3'] + \
                           [f'sensor_{i}' for i in range(1, 22)]
        
    def load_dataset(self, dataset_name='FD001'):
        train_file = f'{self.data_dir}/train_{dataset_name}.txt'
        test_file = f'{self.data_dir}/test_{dataset_name}.txt'
        rul_file = f'{self.data_dir}/RUL_{dataset_name}.txt'
        
        train_df = pd.read_csv(train_file, sep='\s+', header=None, names=self.column_names)
        test_df = pd.read_csv(test_file, sep='\s+', header=None, names=self.column_names)
        rul_values = pd.read_csv(rul_file, sep='\s+', header=None, names=['RUL']).values.flatten()
        
        return train_df, test_df, rul_values
    
    def add_rul_to_train(self, train_df):
        max_cycles = train_df.groupby('unit')['cycle'].max().reset_index()
        max_cycles.columns = ['unit', 'max_cycle']
        
        train_df = train_df.merge(max_cycles, on='unit')
        train_df['RUL'] = train_df['max_cycle'] - train_df['cycle']
        
        train_df['RUL_clip'] = train_df['RUL'].clip(upper=130)
        
        train_df = train_df.drop('max_cycle', axis=1)
        return train_df
    
    def add_rul_to_test(self, test_df, rul_values):
        max_cycles_test = test_df.groupby('unit')['cycle'].max().reset_index()
        max_cycles_test.columns = ['unit', 'max_cycle']
        max_cycles_test['RUL_true'] = rul_values
        max_cycles_test['total_cycle'] = max_cycles_test['max_cycle'] + max_cycles_test['RUL_true']
        
        test_df = test_df.merge(max_cycles_test[['unit', 'total_cycle']], on='unit')
        test_df['RUL'] = test_df['total_cycle'] - test_df['cycle']
        test_df['RUL_clip'] = test_df['RUL'].clip(upper=130)
        test_df = test_df.drop('total_cycle', axis=1)
        
        return test_df
    
    def normalize(self, train_df, test_df, method='standard'):
        feature_cols = [c for c in train_df.columns 
                       if c not in ['unit', 'cycle', 'RUL', 'RUL_clip']]
        
        if method == 'standard':
            scaler = StandardScaler()
        else:
            scaler = MinMaxScaler()
        
        train_df[feature_cols] = scaler.fit_transform(train_df[feature_cols])
        test_df[feature_cols] = scaler.transform(test_df[feature_cols])
        
        return train_df, test_df, scaler
    
    def remove_constant_features(self, df):
        feature_cols = [c for c in df.columns 
                       if c not in ['unit', 'cycle', 'RUL', 'RUL_clip']]
        
        stds = df[feature_cols].std()
        constant_cols = stds[stds < 1e-10].index.tolist()
        
        if constant_cols:
            df = df.drop(columns=constant_cols)
            print(f"移除恒定特征: {constant_cols}")
        
        return df
    
    def prepare_sequence_data(self, df, seq_length=30, feature_cols=None):
        if feature_cols is None:
            feature_cols = [c for c in df.columns 
                           if c not in ['unit', 'cycle', 'RUL', 'RUL_clip']]
        
        sequences = []
        labels = []
        units = []
        
        for unit_id in df['unit'].unique():
            unit_data = df[df['unit'] == unit_id].sort_values('cycle')
            unit_values = unit_data[feature_cols].values
            unit_labels = unit_data['RUL_clip'].values
            
            for i in range(len(unit_values) - seq_length + 1):
                seq = unit_values[i:i+seq_length]
                label = unit_labels[i+seq_length-1]
                sequences.append(seq)
                labels.append(label)
                units.append(unit_id)
        
        return np.array(sequences), np.array(labels), np.array(units)

class NCMAPSSLoader:
    
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
    
    def load_dataset(self, dataset_name='N-CMAPSS_DS01-005'):
        file_path = f'{self.data_dir}/{dataset_name}.h5'
        
        with h5py.File(file_path, 'r') as f:
            data = {}
            for key in ['A_dev', 'A_test', 'W_dev', 'W_test', 'X_s_dev', 'X_s_test',
                       'X_v_dev', 'X_v_test', 'T_dev', 'T_test', 'Y_dev', 'Y_test']:
                data[key] = f[key][:]
            
            for key in ['A_var', 'W_var', 'X_s_var', 'X_v_var', 'T_var']:
                data[key] = [v.decode() if isinstance(v, bytes) else v for v in f[key][:]]
        
        return data
    
    def create_dataframe(self, data, split='dev'):
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
    
    def get_degradation_labels(self, df, thresholds=None):
        if thresholds is None:
            thresholds = {
                'HPC': 0.02,
                'Fan': 0.02,
            }
        
        labels = []
        
        for idx, row in df.iterrows():
            hpc_deg = abs(row['HPC_eff_mod'] - 1.0) + abs(row['HPC_flow_mod'] - 1.0)
            fan_deg = abs(row['fan_eff_mod'] - 1.0) + abs(row['fan_flow_mod'] - 1.0)
            
            hpc_fault = hpc_deg > thresholds['HPC']
            fan_fault = fan_deg > thresholds['Fan']
            
            if hpc_fault and fan_fault:
                labels.append(3)
            elif hpc_fault:
                labels.append(2)
            elif fan_fault:
                labels.append(1)
            else:
                labels.append(0)
        
        df['fault_label'] = labels
        return df
    
    def normalize(self, train_df, test_df, method='standard'):
        exclude_cols = ['unit', 'cycle', 'Fc', 'hs', 'RUL', 'fault_label']
        feature_cols = [c for c in train_df.columns if c not in exclude_cols]
        
        if method == 'standard':
            scaler = StandardScaler()
        else:
            scaler = MinMaxScaler()
        
        train_df[feature_cols] = scaler.fit_transform(train_df[feature_cols])
        test_df[feature_cols] = scaler.transform(test_df[feature_cols])
        
        return train_df, test_df, scaler
    
    def prepare_sequence_data(self, df, seq_length=50, feature_cols=None):
        if feature_cols is None:
            exclude_cols = ['unit', 'cycle', 'Fc', 'hs', 'RUL', 'fault_label']
            feature_cols = [c for c in df.columns if c not in exclude_cols]
        
        sequences = []
        labels_rul = []
        labels_fault = []
        units = []
        
        for unit_id in df['unit'].unique():
            unit_data = df[df['unit'] == unit_id].sort_values('cycle')
            unit_values = unit_data[feature_cols].values
            unit_rul = unit_data['RUL'].values
            unit_fault = unit_data['fault_label'].values if 'fault_label' in unit_data.columns else None
            
            for i in range(len(unit_values) - seq_length + 1):
                seq = unit_values[i:i+seq_length]
                sequences.append(seq)
                labels_rul.append(unit_rul[i+seq_length-1])
                if unit_fault is not None:
                    labels_fault.append(unit_fault[i+seq_length-1])
                units.append(unit_id)
        
        result = {
            'sequences': np.array(sequences),
            'rul_labels': np.array(labels_rul),
            'units': np.array(units)
        }
        
        if labels_fault:
            result['fault_labels'] = np.array(labels_fault)
        
        return result

class TurbofanDataset(Dataset):
    
    def __init__(self, sequences, labels):
        self.sequences = torch.FloatTensor(sequences)
        self.labels = torch.FloatTensor(labels)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]

def create_dataloaders(train_seq, train_labels, test_seq, test_labels, 
                      batch_size=64, val_split=0.1):
    
    if val_split > 0:
        train_seq, val_seq, train_labels, val_labels = train_test_split(
            train_seq, train_labels, test_size=val_split, random_state=42
        )
        val_dataset = TurbofanDataset(val_seq, val_labels)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    else:
        val_loader = None
    
    train_dataset = TurbofanDataset(train_seq, train_labels)
    test_dataset = TurbofanDataset(test_seq, test_labels)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader

if __name__ == '__main__':
    print("=" * 60)
    print("C-MAPSS数据集加载测试")
    print("=" * 60)
    
    cmapss = CMAPSSLoader('./C_MAPSS')
    train_df, test_df, rul_values = cmapss.load_dataset('FD001')
    
    print(f"训练集形状: {train_df.shape}")
    print(f"测试集形状: {test_df.shape}")
    print(f"RUL值数量: {len(rul_values)}")
    print(f"训练集发动机数量: {train_df['unit'].nunique()}")
    print(f"测试集发动机数量: {test_df['unit'].nunique()}")
    
    train_df = cmapss.add_rul_to_train(train_df)
    test_df = cmapss.add_rul_to_test(test_df, rul_values)
    
    print(f"\n训练集RUL统计: min={train_df['RUL'].min()}, max={train_df['RUL'].max()}, mean={train_df['RUL'].mean():.2f}")
    print(f"测试集RUL统计: min={test_df['RUL'].min()}, max={test_df['RUL'].max()}, mean={test_df['RUL'].mean():.2f}")
    
    train_df = cmapss.remove_constant_features(train_df)
    test_df = cmapss.remove_constant_features(test_df)
    
    train_df, test_df, scaler = cmapss.normalize(train_df, test_df)
    
    train_seq, train_labels, train_units = cmapss.prepare_sequence_data(train_df, seq_length=30)
    test_seq, test_labels, test_units = cmapss.prepare_sequence_data(test_df, seq_length=30)
    
    print(f"\n序列数据形状:")
    print(f"  训练集: {train_seq.shape}, 标签: {train_labels.shape}")
    print(f"  测试集: {test_seq.shape}, 标签: {test_labels.shape}")
    
    print("\n" + "=" * 60)
    print("N-CMAPSS数据集加载测试")
    print("=" * 60)
    
    ncmapss = NCMAPSSLoader('./N_MAPSS')
    data = ncmapss.load_dataset('N-CMAPSS_DS01-005')
    
    print(f"数据集变量名:")
    for key in ['A_var', 'W_var', 'X_s_var', 'X_v_var', 'T_var']:
        print(f"  {key}: {data[key]}")
    
    print(f"\n数据形状:")
    for key in ['A_dev', 'W_dev', 'X_s_dev', 'X_v_dev', 'T_dev', 'Y_dev']:
        print(f"  {key}: {data[key].shape}")
    
    train_df_n = ncmapss.create_dataframe(data, split='dev')
    test_df_n = ncmapss.create_dataframe(data, split='test')
    
    print(f"\n开发集DataFrame形状: {train_df_n.shape}")
    print(f"测试集DataFrame形状: {test_df_n.shape}")
    print(f"开发集发动机数量: {train_df_n['unit'].nunique()}")
    print(f"测试集发动机数量: {test_df_n['unit'].nunique()}")
    
    train_df_n = ncmapss.get_degradation_labels(train_df_n)
    test_df_n = ncmapss.get_degradation_labels(test_df_n)
    
    print(f"\n故障类型分布:")
    print(train_df_n['fault_label'].value_counts().sort_index())
    
    print("\n数据加载测试完成!")
