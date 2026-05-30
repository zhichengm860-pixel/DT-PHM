
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import json
import os
import time

torch.backends.cudnn.enabled = False

from data_loader import CMAPSSLoader, create_dataloaders
from models import LSTMModel
from train_engine import Trainer

class PhysicsInformedLoss(nn.Module):
    
    def __init__(self, lambda_physics=1.0):
        super().__init__()
        self.lambda_physics = lambda_physics
    
    def forward(self, pred, target, severity=None):
        mse_loss = nn.MSELoss()(pred, target)
        
        physics_loss = 0
        
        boundary_loss = torch.mean(torch.relu(-pred))
        
        exp_constraint = torch.mean(torch.relu(pred - 130))
        
        physics_loss = boundary_loss + exp_constraint
        
        return mse_loss + self.lambda_physics * physics_loss

class AttentionFusion(nn.Module):
    
    def __init__(self, hidden_dim=32):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 2),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, physics_pred, data_pred):
        stacked = torch.stack([physics_pred, data_pred], dim=-1)
        weights = self.attention(stacked)
        
        final_pred = weights[:, 0] * physics_pred + weights[:, 1] * data_pred
        return final_pred, weights

class PhysicsBaselineModel(nn.Module):
    
    def __init__(self, t_dim=10):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(t_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    
    def forward(self, x):
        if x.dim() == 3:
            t_deg = x[:, -1, -10:]
        else:
            t_deg = x[:, -10:]
        return self.model(t_deg).squeeze(-1)

class DT_PHM_CMAPSS(nn.Module):
    
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, 
                 use_attention_fusion=True, lambda_physics=1.0):
        super().__init__()
        self.use_attention_fusion = use_attention_fusion
        self.lambda_physics = lambda_physics
        
        self.physics_model = PhysicsBaselineModel(t_dim=10)
        
        self.data_model = nn.LSTM(
            input_dim, hidden_dim, num_layers,
            batch_first=True, dropout=0.2
        )
        self.data_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1)
        )
        
        if use_attention_fusion:
            self.fusion = AttentionFusion(hidden_dim=32)
        else:
            self.residual_head = nn.Sequential(
                nn.Linear(hidden_dim, 32),
                nn.ReLU(),
                nn.Linear(32, 1)
            )
    
    def forward(self, x):
        t_deg = x[:, -1, -10:]
        physics_pred = self.physics_model(t_deg)
        
        lstm_out, _ = self.data_model(x)
        data_features = lstm_out[:, -1, :]
        
        if self.use_attention_fusion:
            data_pred = self.data_head(data_features).squeeze(-1)
            final_pred, weights = self.fusion(physics_pred, data_pred)
            return final_pred
        else:
            residual = self.residual_head(data_features).squeeze(-1)
            return physics_pred + residual

class TrainerWithPhysics(Trainer):
    
    def __init__(self, model, device='cuda', exp_name='experiment', 
                 lambda_physics=1.0):
        super().__init__(model, device=device, exp_name=exp_name)
        self.lambda_physics = lambda_physics
        self.physics_criterion = PhysicsInformedLoss(lambda_physics)
    
    def train_epoch(self, train_loader, optimizer):
        self.model.train()
        total_loss = 0
        
        for batch in train_loader:
            if len(batch) == 2:
                x, y = batch
                x, y = x.to(self.device), y.to(self.device)
                
                optimizer.zero_grad()
                pred = self.model(x)
                if isinstance(pred, tuple):
                    pred = pred[0]
                
                loss = self.physics_criterion(pred, y)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
        
        return total_loss / len(train_loader)

def run_lambda_sensitivity(device='cuda'):
    print("\n" + "="*80)
    print("子实验：物理约束权重λ敏感性分析")
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
    lambdas = [0, 0.1, 0.5, 1.0, 2.0, 5.0]
    results = {}
    
    for lam in lambdas:
        print(f"\n{'='*60}")
        print(f"λ = {lam}")
        print('='*60)
        
        model = DT_PHM_CMAPSS(
            input_dim=input_dim, hidden_dim=64, num_layers=2,
            use_attention_fusion=True, lambda_physics=lam
        )
        model = model.to(device)
        
        if lam > 0:
            trainer = TrainerWithPhysics(
                model, device=device, 
                exp_name=f'exp2_lambda_{lam}',
                lambda_physics=lam
            )
        else:
            trainer = Trainer(model, device=device, exp_name=f'exp2_lambda_{lam}')
        
        trainer.train(train_loader, val_loader, epochs=20, lr=0.001, patience=5)
        
        _, metrics, _, _ = trainer.evaluate(test_loader)
        results[f'lambda_{lam}'] = {
            'RMSE': float(metrics['RMSE']),
            'MAE': float(metrics['MAE']),
            'R2': float(metrics['R2'])
        }
        
        print(f"  RMSE: {metrics['RMSE']:.2f}, MAE: {metrics['MAE']:.2f}, R²: {metrics['R2']:.4f}")
    
    df = pd.DataFrame(results).T
    print("\n" + "="*80)
    print("λ敏感性分析结果")
    print("="*80)
    print(df)
    
    os.makedirs('./experiments', exist_ok=True)
    with open('./experiments/exp2_lambda_sensitivity.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    return results

def run_ablation_cmapss(device='cuda'):
    print("\n" + "="*80)
    print("子实验：C-MAPSS协同建模消融")
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
    
    configs = [
        {'name': 'A_pure_physics', 'model': PhysicsBaselineModel(t_dim=10)},
        {'name': 'B_pure_data', 'model': LSTMModel(input_dim=input_dim, hidden_dim=64, num_layers=2)},
        {'name': 'D_residual_collab', 'model': DT_PHM_CMAPSS(input_dim, use_attention_fusion=False)},
        {'name': 'E_attention_collab', 'model': DT_PHM_CMAPSS(input_dim, use_attention_fusion=True)},
    ]
    
    results = {}
    for config in configs:
        print(f"\n{'='*60}")
        print(f"模型: {config['name']}")
        print('='*60)
        
        model = config['model'].to(device)
        trainer = Trainer(model, device=device, exp_name=f'exp2_{config["name"]}')
        
        start_time = time.time()
        trainer.train(train_loader, val_loader, epochs=20, lr=0.001, patience=5)
        train_time = time.time() - start_time
        
        _, metrics, _, _ = trainer.evaluate(test_loader)
        results[config['name']] = {
            'RMSE': float(metrics['RMSE']),
            'MAE': float(metrics['MAE']),
            'R2': float(metrics['R2']),
            'train_time': train_time
        }
        
        print(f"  RMSE: {metrics['RMSE']:.2f}, MAE: {metrics['MAE']:.2f}, R²: {metrics['R2']:.4f}")
        print(f"  训练时间: {train_time:.1f}s")
    
    df = pd.DataFrame(results).T
    print("\n" + "="*80)
    print("C-MAPSS消融实验结果")
    print("="*80)
    print(df)
    
    os.makedirs('./experiments', exist_ok=True)
    with open('./experiments/exp2_cmapss_ablation.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    return results

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    
    results_ablation = run_ablation_cmapss(device=device)
    results_lambda = run_lambda_sensitivity(device=device)
    
    print("\n" + "="*80)
    print("实验二V3完成!")
    print("="*80)

if __name__ == '__main__':
    main()
