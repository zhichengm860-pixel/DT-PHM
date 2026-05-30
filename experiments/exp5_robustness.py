
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import json
import os
import time
import copy

torch.backends.cudnn.enabled = False

from data_loader import CMAPSSLoader, create_dataloaders
from models import get_model
from train_engine import Trainer, MetricsTracker

def train_and_save_model(dataset_name, model_name, epochs=30, device='cuda',
                         use_physics=False, use_constraint=False):
    loader = CMAPSSLoader('./C_MAPSS')
    train_df, test_df, rul_values = loader.load_dataset(dataset_name)
    train_df = loader.add_rul_to_train(train_df)
    test_df = loader.add_rul_to_test(test_df, rul_values)
    train_df = loader.remove_constant_features(train_df)
    test_df = loader.remove_constant_features(test_df)
    train_df, test_df, scaler = loader.normalize(train_df, test_df)
    
    train_seq, train_labels, _ = loader.prepare_sequence_data(train_df, seq_length=30)
    test_seq, test_labels, _ = loader.prepare_sequence_data(test_df, seq_length=30)
    
    train_loader, val_loader, test_loader = create_dataloaders(
        train_seq, train_labels, test_seq, test_labels,
        batch_size=64, val_split=0.1
    )
    
    input_dim = train_seq.shape[-1]
    if model_name == 'dt_phm':
        model = get_model(model_name, input_dim, t_dim=10, w_dim=4,
                         use_physics=use_physics, use_constraint=use_constraint)
    else:
        model = get_model(model_name, input_dim)
    
    exp_name = f"exp5_{dataset_name}_{model_name}"
    trainer = Trainer(model, device=device, exp_name=exp_name)
    trainer.train(train_loader, val_loader, epochs=epochs, lr=0.001,
                 use_physics_constraint=use_constraint)
    
    clean_metrics, _, _ = trainer.test(test_loader)
    
    return {
        'model': model,
        'trainer': trainer,
        'test_seq': test_seq,
        'test_labels': test_labels,
        'test_loader': test_loader,
        'clean_metrics': clean_metrics,
        'input_dim': input_dim,
        'device': device
    }

def evaluate_with_noise(model, test_seq, test_labels, snr_db, device='cuda',
                        batch_size=256, use_physics=False):
    if snr_db is None:
        noisy_seq = test_seq.copy()
    else:
        signal_power = np.mean(test_seq ** 2)
        noise_power = signal_power / (10 ** (snr_db / 10))
        noise = np.random.normal(0, np.sqrt(noise_power), test_seq.shape)
        noisy_seq = test_seq + noise
    
    dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(noisy_seq),
        torch.FloatTensor(test_labels).unsqueeze(-1)
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    model.eval()
    all_preds = []
    all_true = []
    
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            if use_physics and hasattr(model, 'use_physics') and model.use_physics:
                t_dim = getattr(model, 't_dim', 0)
                if t_dim > 0 and x.shape[-1] >= t_dim:
                    t_deg = x[:, -1, -t_dim:]
                    pred = model(x, t_deg)
                else:
                    pred = model(x)
            else:
                pred = model(x)
            if isinstance(pred, tuple):
                pred = pred[0]
            all_preds.extend(pred.cpu().numpy())
            all_true.extend(y.cpu().numpy())
    
    all_preds = np.array(all_preds).flatten()
    all_true = np.array(all_true).flatten()
    
    return MetricsTracker.compute_all(all_true, all_preds)

def run_noise_robustness(datasets=['FD001', 'FD002'], 
                         snr_levels=[None, 20, 10, 5, 0, -5],
                         epochs=30, device='cuda'):
    print("\n" + "="*80)
    print("子实验5.1: 噪声鲁棒性测试")
    print("="*80)
    
    results = {}
    
    for dataset_name in datasets:
        results[dataset_name] = {}
        
        for model_name in ['lstm', 'dt_phm']:
            use_physics = (model_name == 'dt_phm')
            use_constraint = (model_name == 'dt_phm')
            
            print(f"\n训练 {model_name} on {dataset_name}...")
            data = train_and_save_model(
                dataset_name, model_name, epochs=epochs, device=device,
                use_physics=use_physics, use_constraint=use_constraint
            )
            
            clean_rmse = data['clean_metrics']['RMSE']
            results[dataset_name][model_name] = {
                'clean': {k: float(v) for k, v in data['clean_metrics'].items()}
            }
            
            for snr in snr_levels:
                snr_str = 'clean' if snr is None else f'{snr}dB'
                if snr is None:
                    continue
                
                noisy_metrics = evaluate_with_noise(
                    data['model'], data['test_seq'], data['test_labels'],
                    snr, device=device, use_physics=use_physics
                )
                
                degradation = (noisy_metrics['RMSE'] - clean_rmse) / clean_rmse * 100
                
                results[dataset_name][model_name][snr_str] = {
                    'RMSE': float(noisy_metrics['RMSE']),
                    'MAE': float(noisy_metrics['MAE']),
                    'Score': float(noisy_metrics['Score']),
                    'R2': float(noisy_metrics['R2']),
                    'degradation_%': float(degradation)
                }
                
                print(f"  {snr_str}: RMSE={noisy_metrics['RMSE']:.2f} (退化{degradation:.1f}%)")
    
    save_results(results, 'exp5_1_noise_robustness')
    return results

def run_condition_mutation(datasets=['FD002', 'FD004'], epochs=30, device='cuda'):
    print("\n" + "="*80)
    print("子实验5.2: 工况突变测试")
    print("="*80)
    
    results = {}
    
    for dataset_name in datasets:
        results[dataset_name] = {}
        
        for model_name in ['lstm', 'dt_phm']:
            use_physics = (model_name == 'dt_phm')
            use_constraint = (model_name == 'dt_phm')
            
            metrics, _, _ = run_experiment_simple(
                dataset_name, model_name, epochs=epochs, device=device,
                use_physics=use_physics, use_constraint=use_constraint
            )
            results[dataset_name][model_name] = {k: float(v) for k, v in metrics.items()}
    
    save_results(results, 'exp5_2_condition_mutation')
    return results

def run_experiment_simple(dataset_name, model_name, epochs=30, device='cuda',
                          use_physics=False, use_constraint=False):
    loader = CMAPSSLoader('./C_MAPSS')
    train_df, test_df, rul_values = loader.load_dataset(dataset_name)
    train_df = loader.add_rul_to_train(train_df)
    test_df = loader.add_rul_to_test(test_df, rul_values)
    train_df = loader.remove_constant_features(train_df)
    test_df = loader.remove_constant_features(test_df)
    train_df, test_df, scaler = loader.normalize(train_df, test_df)
    
    train_seq, train_labels, _ = loader.prepare_sequence_data(train_df, seq_length=30)
    test_seq, test_labels, _ = loader.prepare_sequence_data(test_df, seq_length=30)
    
    train_loader, val_loader, test_loader = create_dataloaders(
        train_seq, train_labels, test_seq, test_labels,
        batch_size=64, val_split=0.1
    )
    
    input_dim = train_seq.shape[-1]
    if model_name == 'dt_phm':
        model = get_model(model_name, input_dim, t_dim=10, w_dim=4,
                         use_physics=use_physics, use_constraint=use_constraint)
    else:
        model = get_model(model_name, input_dim)
    
    exp_name = f"exp5_{dataset_name}_{model_name}"
    trainer = Trainer(model, device=device, exp_name=exp_name)
    trainer.train(train_loader, val_loader, epochs=epochs, lr=0.001,
                 use_physics_constraint=use_constraint)
    
    metrics, _, _ = trainer.test(test_loader)
    return metrics, model, test_seq, test_labels

def run_cross_dataset(epochs=30, device='cuda'):
    print("\n" + "="*80)
    print("子实验5.3: 跨数据集泛化测试")
    print("="*80)
    
    results = {}
    
    for model_name in ['lstm', 'dt_phm']:
        use_physics = (model_name == 'dt_phm')
        use_constraint = (model_name == 'dt_phm')
        
        print(f"\n在FD001上训练 {model_name}...")
        train_loader_fd001 = CMAPSSLoader('./C_MAPSS')
        train_df, _, _ = train_loader_fd001.load_dataset('FD001')
        train_df = train_loader_fd001.add_rul_to_train(train_df)
        train_df = train_loader_fd001.remove_constant_features(train_df)
        
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        feature_cols = [c for c in train_df.columns if c not in ['unit', 'cycle', 'RUL']]
        train_df[feature_cols] = scaler.fit_transform(train_df[feature_cols])
        
        train_seq, train_labels, _ = train_loader_fd001.prepare_sequence_data(train_df, seq_length=30)
        
        input_dim = train_seq.shape[-1]
        if model_name == 'dt_phm':
            model = get_model(model_name, input_dim, t_dim=10, w_dim=4,
                             use_physics=use_physics, use_constraint=use_constraint)
        else:
            model = get_model(model_name, input_dim)
        
        dataset = torch.utils.data.TensorDataset(
            torch.FloatTensor(train_seq),
            torch.FloatTensor(train_labels).unsqueeze(-1)
        )
        train_size = int(0.9 * len(dataset))
        val_size = len(dataset) - train_size
        train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])
        train_l = torch.utils.data.DataLoader(train_ds, batch_size=64, shuffle=True)
        val_l = torch.utils.data.DataLoader(val_ds, batch_size=64, shuffle=False)
        
        exp_name = f"exp5_cross_{model_name}"
        trainer = Trainer(model, device=device, exp_name=exp_name)
        trainer.train(train_l, val_l, epochs=epochs, lr=0.001,
                     use_physics_constraint=use_constraint)
        
        test_loader_fd002 = CMAPSSLoader('./C_MAPSS')
        test_df, test_rul_df, rul_values = test_loader_fd002.load_dataset('FD002')
        test_df = test_loader_fd002.add_rul_to_test(test_df, rul_values)
        test_df = test_loader_fd002.remove_constant_features(test_df)
        
        fd002_feature_cols = [c for c in test_df.columns if c not in ['unit', 'cycle', 'RUL']]
        
        if len(fd002_feature_cols) != input_dim:
            print(f"  特征维度不匹配: FD001={input_dim}, FD002={len(fd002_feature_cols)}，跳过")
            results[model_name] = {'error': 'feature_dim_mismatch'}
            continue
        
        test_df_fd002 = test_df.copy()
        test_df_fd002[fd002_feature_cols] = scaler.transform(test_df_fd002[fd002_feature_cols])
        
        test_seq, test_labels, _ = test_loader_fd002.prepare_sequence_data(test_df_fd002, seq_length=30)
        
        test_dataset = torch.utils.data.TensorDataset(
            torch.FloatTensor(test_seq),
            torch.FloatTensor(test_labels).unsqueeze(-1)
        )
        test_l = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)
        
        metrics, _, _ = trainer.test(test_l)
        results[model_name] = {k: float(v) for k, v in metrics.items()}
        print(f"  {model_name} on FD002: RMSE={metrics['RMSE']:.2f}")
    
    save_results(results, 'exp5_3_cross_dataset')
    return results

def save_results(results, filename):
    os.makedirs('./experiments', exist_ok=True)
    
    def convert_numpy(obj):
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_numpy(i) for i in obj]
        return obj
    
    with open(f'./experiments/{filename}.json', 'w') as f:
        json.dump(convert_numpy(results), f, indent=2)
    
    rows = []
    for dataset, models in results.items():
        for model, scenarios in models.items():
            if isinstance(scenarios, dict):
                for scenario, metrics in scenarios.items():
                    if isinstance(metrics, dict) and 'error' not in metrics:
                        row = {'Dataset': dataset, 'Model': model, 'Scenario': scenario}
                        row.update({k: v for k, v in metrics.items() if isinstance(v, (int, float))})
                        rows.append(row)
    
    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(f'./experiments/{filename}.csv', index=False)
        print(f"\n结果汇总: {filename}")
        print(df.to_string(index=False))

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    
    noise_results = run_noise_robustness(
        datasets=['FD001', 'FD002'],
        snr_levels=[20, 10, 5, 0, -5],
        epochs=30,
        device=device
    )
    
    mutation_results = run_condition_mutation(
        datasets=['FD002', 'FD004'],
        epochs=30,
        device=device
    )
    
    cross_results = run_cross_dataset(
        epochs=30,
        device=device
    )
    
    print("\n" + "="*80)
    print("实验五完成!")
    print("="*80)

if __name__ == '__main__':
    main()
