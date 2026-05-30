
import torch
import numpy as np
import pandas as pd
import json
import os
import sys
import time

torch.backends.cudnn.enabled = False

from data_loader import CMAPSSLoader, create_dataloaders
from models import get_model
from train_engine import Trainer

def run_experiment(dataset_name='FD001', model_name='lstm', epochs=50, 
                   seq_length=30, batch_size=64, lr=0.001, device='cuda',
                   use_physics=False, use_constraint=False):
    print(f"\n{'='*70}")
    print(f"数据集: {dataset_name}, 模型: {model_name}")
    print(f"物理模型: {use_physics}, 物理约束: {use_constraint}")
    print(f"{'='*70}")
    
    loader = CMAPSSLoader('./C_MAPSS')
    train_df, test_df, rul_values = loader.load_dataset(dataset_name)
    
    train_df = loader.add_rul_to_train(train_df)
    test_df = loader.add_rul_to_test(test_df, rul_values)
    
    train_df = loader.remove_constant_features(train_df)
    test_df = loader.remove_constant_features(test_df)
    
    train_df, test_df, scaler = loader.normalize(train_df, test_df)
    
    train_seq, train_labels, _ = loader.prepare_sequence_data(train_df, seq_length=seq_length)
    test_seq, test_labels, _ = loader.prepare_sequence_data(test_df, seq_length=seq_length)
    
    print(f"训练样本: {len(train_seq)}, 测试样本: {len(test_seq)}")
    print(f"特征维度: {train_seq.shape[-1]}")
    
    train_loader, val_loader, test_loader = create_dataloaders(
        train_seq, train_labels, test_seq, test_labels,
        batch_size=batch_size, val_split=0.1
    )
    
    input_dim = train_seq.shape[-1]
    
    if model_name == 'dt_phm':
        model = get_model(model_name, input_dim, 
                         t_dim=10, w_dim=4,
                         use_physics=use_physics, 
                         use_constraint=use_constraint)
    else:
        model = get_model(model_name, input_dim)
    
    exp_name = f"exp4_{dataset_name}_{model_name}"
    if use_physics:
        exp_name += "_physics"
    if use_constraint:
        exp_name += "_constraint"
    
    trainer = Trainer(model, device=device, exp_name=exp_name)
    
    start_time = time.time()
    trainer.train(train_loader, val_loader, epochs=epochs, lr=lr,
                 use_physics_constraint=use_constraint)
    train_time = time.time() - start_time
    
    metrics, preds, true = trainer.test(test_loader)
    
    metrics['train_time'] = train_time
    
    print(f"训练时间: {train_time:.1f}s")
    
    return metrics, preds, true

def run_all_baseline_experiments(datasets=['FD001', 'FD002', 'FD003', 'FD004'],
                                 models=['lstm', 'bilstm', 'gru', 'cnn_lstm', 'transformer'],
                                 epochs=50, device='cuda'):
    results = {}
    
    for dataset in datasets:
        results[dataset] = {}
        for model_name in models:
            try:
                metrics, _, _ = run_experiment(
                    dataset_name=dataset,
                    model_name=model_name,
                    epochs=epochs,
                    device=device
                )
                results[dataset][model_name] = metrics
            except Exception as e:
                print(f"错误: {dataset}-{model_name}: {e}")
                results[dataset][model_name] = {'error': str(e)}
    
    save_results_table(results, 'exp4_baseline_comparison')
    
    return results

def run_single_experiment(dataset_name='FD001', model_name='lstm', epochs=30, device='cuda'):
    print(f"\n{'='*70}")
    print(f"快速实验: {dataset_name} - {model_name}")
    print(f"{'='*70}")
    
    metrics, preds, true = run_experiment(
        dataset_name=dataset_name,
        model_name=model_name,
        epochs=epochs,
        device=device
    )
    
    return metrics

def run_dt_phm_ablation(datasets=['FD001', 'FD002', 'FD003', 'FD004'],
                        epochs=50, device='cuda'):
    configs = [
        {'name': 'dt_phm_data_only', 'use_physics': False, 'use_constraint': False},
        {'name': 'dt_phm_physics', 'use_physics': True, 'use_constraint': False},
        {'name': 'dt_phm_full', 'use_physics': True, 'use_constraint': True},
    ]
    
    results = {}
    
    for dataset in datasets:
        results[dataset] = {}
        for config in configs:
            try:
                metrics, _, _ = run_experiment(
                    dataset_name=dataset,
                    model_name='dt_phm',
                    epochs=epochs,
                    device=device,
                    use_physics=config['use_physics'],
                    use_constraint=config['use_constraint']
                )
                results[dataset][config['name']] = metrics
            except Exception as e:
                print(f"错误: {dataset}-{config['name']}: {e}")
                results[dataset][config['name']] = {'error': str(e)}
    
    save_results_table(results, 'exp4_dt_phm_ablation')
    
    return results

def save_results_table(results, filename):
    rows = []
    for dataset, models in results.items():
        for model, metrics in models.items():
            if 'error' not in metrics:
                row = {
                    'Dataset': dataset,
                    'Model': model,
                    'RMSE': metrics.get('RMSE', np.nan),
                    'MAE': metrics.get('MAE', np.nan),
                    'Score': metrics.get('Score', np.nan),
                    'R2': metrics.get('R2', np.nan),
                    'TrainTime(s)': metrics.get('train_time', np.nan)
                }
                rows.append(row)
    
    df = pd.DataFrame(rows)
    
    os.makedirs('./experiments', exist_ok=True)
    df.to_csv(f'./experiments/{filename}.csv', index=False)
    
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
    
    print(f"\n{'='*80}")
    print(f"结果汇总: {filename}")
    print(f"{'='*80}")
    print(df.to_string(index=False))
    
    print(f"\n{'='*80}")
    print("按数据集分组的RMSE对比")
    print(f"{'='*80}")
    for dataset in df['Dataset'].unique():
        subset = df[df['Dataset'] == dataset][['Model', 'RMSE', 'MAE', 'Score']]
        print(f"\n{dataset}:")
        print(subset.to_string(index=False))

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    
    print("\n" + "="*80)
    print("实验4.1: 基线方法全面对比")
    print("="*80)
    
    baseline_results = run_all_baseline_experiments(
        datasets=['FD001', 'FD002', 'FD003', 'FD004'],
        models=['lstm', 'bilstm', 'gru', 'cnn_lstm', 'transformer'],
        epochs=30,
        device=device
    )
    
    print("\n" + "="*80)
    print("实验4.2/2.1: DT-PHM协同建模消融")
    print("="*80)
    
    dt_results = run_dt_phm_ablation(
        datasets=['FD001', 'FD002', 'FD003', 'FD004'],
        epochs=30,
        device=device
    )
    
    print("\n" + "="*80)
    print("实验四完成! 所有结果保存在 ./experiments/ 目录")
    print("="*80)

if __name__ == '__main__':
    main()
