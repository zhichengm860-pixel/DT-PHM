
import torch
import json
import os
import time

torch.backends.cudnn.enabled = False

from exp4_rul_prediction import run_experiment
from train_engine import MetricsTracker

def run_priority1_experiments(device='cuda'):
    print("\n" + "="*80)
    print("优先级1: RUL预测全面对比 + DT-PHM消融")
    print("="*80)
    
    results = {}
    datasets = ['FD001', 'FD002', 'FD003', 'FD004']
    models = ['lstm', 'bilstm', 'gru', 'cnn_lstm', 'transformer']
    
    for dataset in datasets:
        results[dataset] = {}
        for model_name in models:
            try:
                metrics, _, _ = run_experiment(
                    dataset_name=dataset,
                    model_name=model_name,
                    epochs=30,
                    device=device
                )
                results[dataset][model_name] = {
                    'RMSE': float(metrics['RMSE']),
                    'MAE': float(metrics['MAE']),
                    'Score': float(metrics['Score']),
                    'R2': float(metrics['R2'])
                }
            except Exception as e:
                print(f"错误: {dataset}-{model_name}: {e}")
                results[dataset][model_name] = {'error': str(e)}
    
    os.makedirs('./experiments', exist_ok=True)
    with open('./experiments/priority1_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "="*80)
    print("优先级1结果汇总")
    print("="*80)
    
    import pandas as pd
    rows = []
    for dataset, models in results.items():
        for model, metrics in models.items():
            if 'error' not in metrics:
                rows.append({
                    'Dataset': dataset,
                    'Model': model,
                    'RMSE': metrics['RMSE'],
                    'MAE': metrics['MAE'],
                    'Score': metrics['Score'],
                    'R2': metrics['R2']
                })
    
    df = pd.DataFrame(rows)
    df.to_csv('./experiments/priority1_results.csv', index=False)
    print(df.to_string(index=False))
    
    return results

def run_priority2_experiments(device='cuda'):
    print("\n" + "="*80)
    print("优先级2: 鲁棒性验证")
    print("="*80)
    
    results = {}
    
    for dataset in ['FD001', 'FD002']:
        results[dataset] = {}
        for model_name in ['lstm', 'dt_phm']:
            try:
                metrics, _, _ = run_experiment(
                    dataset_name=dataset,
                    model_name=model_name,
                    epochs=30,
                    device=device,
                    use_physics=(model_name == 'dt_phm'),
                    use_constraint=(model_name == 'dt_phm')
                )
                results[dataset][model_name] = {
                    'RMSE': float(metrics['RMSE']),
                    'MAE': float(metrics['MAE']),
                    'Score': float(metrics['Score']),
                    'R2': float(metrics['R2'])
                }
            except Exception as e:
                print(f"错误: {dataset}-{model_name}: {e}")
                results[dataset][model_name] = {'error': str(e)}
    
    with open('./experiments/priority2_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n优先级2结果:")
    for dataset, models in results.items():
        print(f"\n{dataset}:")
        for model, metrics in models.items():
            if 'error' not in metrics:
                print(f"  {model}: RMSE={metrics['RMSE']:.2f}, Score={metrics['Score']:.2f}")
    
    return results

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    print(f"开始时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    start = time.time()
    priority1_results = run_priority1_experiments(device)
    print(f"\n优先级1耗时: {(time.time()-start)/60:.1f}分钟")
    
    start = time.time()
    priority2_results = run_priority2_experiments(device)
    print(f"\n优先级2耗时: {(time.time()-start)/60:.1f}分钟")
    
    print("\n" + "="*80)
    print("所有实验完成!")
    print(f"结束时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)

if __name__ == '__main__':
    main()
