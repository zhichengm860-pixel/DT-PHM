
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import json
import os
from collections import deque

torch.backends.cudnn.enabled = False

from data_loader import CMAPSSLoader, create_dataloaders
from models import LSTMModel
from train_engine import Trainer

class EWCRegularizer:
    
    def __init__(self, model, device='cuda', lambda_ewc=1000):
        self.model = model
        self.device = device
        self.lambda_ewc = lambda_ewc
        self.params = {n: p.clone().detach() for n, p in model.named_parameters() if p.requires_grad}
        self.fisher = {n: torch.zeros_like(p) for n, p in model.named_parameters() if p.requires_grad}
    
    def compute_fisher(self, data_loader, num_samples=200):
        self.model.eval()
        for n, p in self.fisher.items():
            self.fisher[n].zero_()
        
        count = 0
        for batch in data_loader:
            if count >= num_samples:
                break
            if len(batch) == 2:
                x, y = batch
                x, y = x.to(self.device), y.to(self.device)
                
                self.model.zero_grad()
                pred = self.model(x)
                if isinstance(pred, tuple):
                    pred = pred[0]
                loss = nn.MSELoss()(pred, y)
                loss.backward()
                
                for n, p in self.model.named_parameters():
                    if p.grad is not None and n in self.fisher:
                        self.fisher[n] += p.grad.data ** 2
                count += x.size(0)
        
        for n in self.fisher:
            self.fisher[n] /= count
    
    def penalty(self, model):
        loss = 0
        for n, p in model.named_parameters():
            if n in self.params and n in self.fisher:
                loss += (self.fisher[n] * (p - self.params[n]) ** 2).sum()
        return self.lambda_ewc * loss
    
    def update_params(self, model):
        self.params = {n: p.clone().detach() for n, p in model.named_parameters() if p.requires_grad}

class ExperienceReplay:
    
    def __init__(self, capacity=500):
        self.buffer = deque(maxlen=capacity)
    
    def add(self, x, y):
        for i in range(x.size(0)):
            self.buffer.append((x[i].cpu().numpy(), y[i].cpu().numpy()))
    
    def sample(self, batch_size):
        if len(self.buffer) < batch_size:
            return None
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        samples = [self.buffer[i] for i in indices]
        x = torch.FloatTensor(np.array([s[0] for s in samples]))
        y = torch.FloatTensor(np.array([s[1] for s in samples]))
        return x, y
    
    def __len__(self):
        return len(self.buffer)

class ClosedLoopFrameworkV2:
    
    def __init__(self, model, device='cuda', update_interval=50,
                 update_strategy='simple', lambda_ewc=1000,
                 replay_capacity=500, replay_ratio=0.3):
        self.model = model
        self.device = device
        self.update_interval = update_interval
        self.update_strategy = update_strategy
        self.replay_ratio = replay_ratio
        
        self.ewc = None
        if 'ewc' in update_strategy:
            self.ewc = EWCRegularizer(model, device, lambda_ewc)
        
        self.replay = None
        if 'replay' in update_strategy:
            self.replay = ExperienceReplay(capacity=replay_capacity)
    
    def initialize_ewc(self, data_loader):
        if self.ewc is not None:
            print("  计算Fisher信息矩阵...")
            self.ewc.compute_fisher(data_loader)
            self.ewc.update_params(self.model)
            print("  EWC初始化完成")
    
    def predict_with_update(self, data_loader, epochs_per_update=3):
        predictions = []
        true_values = []
        update_points = []
        buffer_x = []
        buffer_y = []
        batch_count = 0
        
        for batch in data_loader:
            if len(batch) == 2:
                x, y = batch
                x, y = x.to(self.device), y.to(self.device)
                
                self.model.eval()
                with torch.no_grad():
                    pred = self.model(x)
                    if isinstance(pred, tuple):
                        pred = pred[0]
                
                predictions.extend(pred.cpu().numpy().flatten().tolist())
                true_values.extend(y.cpu().numpy().flatten().tolist())
                
                buffer_x.append(x.detach())
                buffer_y.append(y.detach())
                
                if self.replay is not None:
                    self.replay.add(x.detach(), y.detach())
                
                batch_count += 1
                
                if batch_count % self.update_interval == 0:
                    update_points.append(len(predictions))
                    self._online_update(buffer_x, buffer_y, epochs_per_update)
                    buffer_x = []
                    buffer_y = []
        
        return np.array(predictions), np.array(true_values), update_points
    
    def _online_update(self, buffer_x, buffer_y, epochs):
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.00001)
        criterion = nn.MSELoss()
        
        for _ in range(epochs):
            for x, y in zip(buffer_x, buffer_y):
                optimizer.zero_grad()
                pred = self.model(x)
                if isinstance(pred, tuple):
                    pred = pred[0]
                loss = criterion(pred, y)
                
                if self.ewc is not None:
                    ewc_loss = self.ewc.penalty(self.model)
                    loss += ewc_loss
                
                loss.backward()
                optimizer.step()
            
            if self.replay is not None and len(self.replay) > 32:
                replay_batch_size = max(16, int(len(buffer_x[0]) * self.replay_ratio))
                replay_data = self.replay.sample(replay_batch_size)
                if replay_data is not None:
                    rx, ry = replay_data
                    rx, ry = rx.to(self.device), ry.to(self.device)
                    optimizer.zero_grad()
                    pred = self.model(rx)
                    if isinstance(pred, tuple):
                        pred = pred[0]
                    loss = criterion(pred, ry)
                    
                    if self.ewc is not None:
                        ewc_loss = self.ewc.penalty(self.model)
                        loss += ewc_loss
                    
                    loss.backward()
                    optimizer.step()
        
        self.model.eval()

def run_closed_loop_experiment_v2(device='cuda'):
    print("\n" + "="*80)
    print("实验一v2：闭环框架验证 (优化版) (C-MAPSS FD001)")
    print("="*80)
    
    loader = CMAPSSLoader(data_dir='./C_MAPSS')
    train_df, test_df, rul_values = loader.load_dataset('FD001')
    
    train_df = loader.add_rul_to_train(train_df)
    test_df = loader.add_rul_to_test(test_df, rul_values)
    
    train_df, test_df, _ = loader.normalize(train_df, test_df)
    
    train_X, train_y, _ = loader.prepare_sequence_data(train_df, seq_length=30)
    test_X, test_y, _ = loader.prepare_sequence_data(test_df, seq_length=30)
    
    train_loader, val_loader, test_loader = create_dataloaders(
        train_X, train_y, test_X, test_y, batch_size=64
    )
    
    input_dim = train_X.shape[2]
    results = {}
    
    methods = [
        ('open_loop', None),
        ('closed_loop_simple', 'simple'),
        ('closed_loop_ewc', 'ewc'),
        ('closed_loop_replay', 'replay'),
        ('closed_loop_ewc_replay', 'ewc_replay'),
    ]
    
    for method_name, strategy in methods:
        print(f"\n{'='*60}")
        print(f"方法: {method_name}")
        print('='*60)
        
        model = LSTMModel(input_dim=input_dim, hidden_dim=64, num_layers=2)
        model = model.to(device)
        
        trainer = Trainer(model, device=device, exp_name=f'exp1v2_{method_name}')
        trainer.train(train_loader, val_loader, epochs=30, lr=0.001, patience=7)
        
        if strategy is None:
            _, test_metrics, _, _ = trainer.evaluate(test_loader)
            results[method_name] = {
                'RMSE': float(test_metrics['RMSE']),
                'MAE': float(test_metrics['MAE']),
                'R2': float(test_metrics['R2'])
            }
        else:
            framework = ClosedLoopFrameworkV2(
                model, device=device, update_interval=50,
                update_strategy=strategy, lambda_ewc=1000,
                replay_capacity=500, replay_ratio=0.3
            )
            
            if 'ewc' in strategy:
                framework.initialize_ewc(train_loader)
            
            predictions, true_values, update_points = framework.predict_with_update(
                test_loader, epochs_per_update=3
            )
            
            rmse = np.sqrt(np.mean((predictions - true_values) ** 2))
            mae = np.mean(np.abs(predictions - true_values))
            r2 = 1 - np.sum((predictions - true_values) ** 2) / np.sum((true_values - np.mean(true_values)) ** 2)
            
            results[method_name] = {
                'RMSE': float(rmse),
                'MAE': float(mae),
                'R2': float(r2),
                'update_points': len(update_points)
            }
        
        print(f"  RMSE: {results[method_name]['RMSE']:.2f}")
        print(f"  MAE: {results[method_name]['MAE']:.2f}")
        print(f"  R²: {results[method_name]['R2']:.4f}")
    
    df = pd.DataFrame(results).T
    print("\n" + "="*80)
    print("结果汇总: exp1_closed_loop_v2")
    print("="*80)
    print(df)
    
    os.makedirs('./experiments', exist_ok=True)
    with open('./experiments/exp1_v2_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    return results

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    
    results = run_closed_loop_experiment_v2(device)
    
    print("\n" + "="*80)
    print("实验一v2完成!")
    print("="*80)

if __name__ == '__main__':
    main()
