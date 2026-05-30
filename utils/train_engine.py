
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import json
import os
import time
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

class MetricsTracker:
    
    @staticmethod
    def rmse(y_true, y_pred):
        return np.sqrt(np.mean((y_true - y_pred) ** 2))
    
    @staticmethod
    def mae(y_true, y_pred):
        return np.mean(np.abs(y_true - y_pred))
    
    @staticmethod
    def score(y_true, y_pred):
        d = y_pred - y_true
        score = np.where(d < 0, 
                        np.exp(-d / 13) - 1,
                        np.exp(d / 10) - 1)
        return np.sum(score)
    
    @staticmethod
    def r2(y_true, y_pred):
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return 1 - ss_res / (ss_tot + 1e-10)
    
    @classmethod
    def compute_all(cls, y_true, y_pred):
        return {
            'RMSE': cls.rmse(y_true, y_pred),
            'MAE': cls.mae(y_true, y_pred),
            'Score': cls.score(y_true, y_pred),
            'R2': cls.r2(y_true, y_pred)
        }

class ExperimentLogger:
    
    def __init__(self, exp_name, save_dir='./experiments'):
        self.exp_name = exp_name
        self.save_dir = os.path.join(save_dir, exp_name)
        os.makedirs(self.save_dir, exist_ok=True)
        
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_metrics': [],
            'test_metrics': None,
            'config': {},
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
    
    def log_config(self, config):
        self.history['config'] = config
        with open(os.path.join(self.save_dir, 'config.json'), 'w') as f:
            json.dump(config, f, indent=2)
    
    def log_epoch(self, epoch, train_loss, val_loss, val_metrics):
        self.history['train_loss'].append(train_loss)
        self.history['val_loss'].append(val_loss)
        self.history['val_metrics'].append(val_metrics)
        
        print(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, "
              f"RMSE={val_metrics['RMSE']:.2f}, Score={val_metrics['Score']:.2f}")
    
    def log_test_results(self, test_metrics, predictions=None, true_values=None):
        self.history['test_metrics'] = test_metrics
        
        serializable_metrics = {}
        for k, v in test_metrics.items():
            if isinstance(v, (np.integer, np.floating, np.int64, np.float32, np.float64)):
                serializable_metrics[k] = float(v)
            elif isinstance(v, torch.Tensor):
                serializable_metrics[k] = float(v.item())
            else:
                serializable_metrics[k] = v
        
        with open(os.path.join(self.save_dir, 'test_metrics.json'), 'w') as f:
            json.dump(serializable_metrics, f, indent=2)
            f.flush()
        
        if predictions is not None and true_values is not None:
            results_df = pd.DataFrame({
                'true': true_values,
                'pred': predictions,
                'error': predictions - true_values
            })
            results_df.to_csv(os.path.join(self.save_dir, 'predictions.csv'), index=False)
    
    def plot_training_curve(self):
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        axes[0].plot(self.history['train_loss'], label='Train Loss')
        axes[0].plot(self.history['val_loss'], label='Val Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training Curve')
        axes[0].legend()
        axes[0].grid(True)
        
        val_rmse = [m['RMSE'] for m in self.history['val_metrics']]
        axes[1].plot(val_rmse, label='Val RMSE', color='red')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('RMSE')
        axes[1].set_title('Validation RMSE')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'training_curve.png'), dpi=150)
        plt.close()
    
    def plot_predictions(self, y_true, y_pred, title='RUL Prediction'):
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        axes[0].scatter(y_true, y_pred, alpha=0.5, s=20)
        axes[0].plot([0, max(y_true)], [0, max(y_true)], 'r--', label='Perfect Prediction')
        axes[0].set_xlabel('True RUL')
        axes[0].set_ylabel('Predicted RUL')
        axes[0].set_title(f'{title}\nR²={MetricsTracker.r2(y_true, y_pred):.3f}')
        axes[0].legend()
        axes[0].grid(True)
        
        residuals = y_pred - y_true
        axes[1].hist(residuals, bins=50, edgecolor='black', alpha=0.7)
        axes[1].axvline(x=0, color='r', linestyle='--')
        axes[1].set_xlabel('Prediction Error')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title(f'Residual Distribution\nMean={np.mean(residuals):.2f}, Std={np.std(residuals):.2f}')
        axes[1].grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'predictions.png'), dpi=150)
        plt.close()

class Trainer:
    
    def __init__(self, model, device='cuda', exp_name='experiment'):
        self.model = model.to(device)
        self.device = device
        self.logger = ExperimentLogger(exp_name)
        self.best_val_loss = float('inf')
        self.best_model_state = None
        self.patience_counter = 0
    
    def train(self, train_loader, val_loader, epochs=100, lr=0.001, 
              weight_decay=1e-5, patience=10, physics_weight=0.1,
              use_physics_constraint=False, t_degradation_fn=None):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
        criterion = nn.MSELoss()
        
        self.logger.log_config({
            'epochs': epochs,
            'lr': lr,
            'weight_decay': weight_decay,
            'patience': patience,
            'physics_weight': physics_weight,
            'use_physics_constraint': use_physics_constraint,
            'model': self.model.__class__.__name__
        })
        
        for epoch in range(epochs):
            self.model.train()
            train_losses = []
            
            for batch in train_loader:
                if len(batch) == 2:
                    x, y = batch
                    x, y = x.to(self.device), y.to(self.device)
                    
                    optimizer.zero_grad()
                    
                    if hasattr(self.model, 'use_physics') and self.model.use_physics:
                        t_dim = getattr(self.model, 't_dim', 0)
                        if t_dim > 0 and x.shape[-1] >= t_dim:
                            t_deg = x[:, -1, -t_dim:]
                            pred = self.model(x, t_deg)
                            if isinstance(pred, tuple):
                                pred = pred[0]
                            
                            loss = criterion(pred, y)
                            
                            if use_physics_constraint:
                                physics_loss = self.model.compute_physics_loss(pred, t_deg)
                                loss += physics_weight * physics_loss
                        else:
                            pred = self.model(x)
                            if isinstance(pred, tuple):
                                pred = pred[0]
                            loss = criterion(pred, y)
                    else:
                        pred = self.model(x)
                        if isinstance(pred, tuple):
                            pred = pred[0]
                        loss = criterion(pred, y)
                    
                    loss.backward()
                    optimizer.step()
                    train_losses.append(loss.item())
            
            avg_train_loss = np.mean(train_losses)
            
            val_loss, val_metrics, _, _ = self.evaluate(val_loader)
            
            self.logger.log_epoch(epoch, avg_train_loss, val_loss, val_metrics)
            
            scheduler.step(val_loss)
            
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_model_state = self.model.state_dict().copy()
                self.patience_counter = 0
            else:
                self.patience_counter += 1
                if self.patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch}")
                    break
        
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
    
    def evaluate(self, data_loader, return_predictions=False):
        self.model.eval()
        all_preds = []
        all_true = []
        total_loss = 0.0
        criterion = nn.MSELoss()
        
        with torch.no_grad():
            for batch in data_loader:
                if len(batch) == 2:
                    x, y = batch
                    x, y = x.to(self.device), y.to(self.device)
                    
                    if hasattr(self.model, 'use_physics') and self.model.use_physics:
                        t_dim = getattr(self.model, 't_dim', 0)
                        if t_dim > 0 and x.shape[-1] >= t_dim:
                            t_deg = x[:, -1, -t_dim:]
                            pred = self.model(x, t_deg)
                        else:
                            pred = self.model(x)
                    else:
                        pred = self.model(x)
                    
                    if isinstance(pred, tuple):
                        pred = pred[0]
                    
                    loss = criterion(pred, y)
                    total_loss += loss.item()
                    
                    all_preds.extend(pred.cpu().numpy().flatten().tolist())
                    all_true.extend(y.cpu().numpy().flatten().tolist())
        
        avg_loss = total_loss / len(data_loader)
        all_preds = np.array(all_preds)
        all_true = np.array(all_true)
        
        metrics = MetricsTracker.compute_all(all_true, all_preds)
        
        if return_predictions:
            return avg_loss, metrics, all_preds, all_true
        return avg_loss, metrics, None, None
    
    def test(self, test_loader):
        loss, metrics, preds, true = self.evaluate(test_loader, return_predictions=True)
        
        print(f"\nTest Results:")
        print(f"  RMSE: {metrics['RMSE']:.2f}")
        print(f"  MAE: {metrics['MAE']:.2f}")
        print(f"  Score: {metrics['Score']:.2f}")
        print(f"  R2: {metrics['R2']:.3f}")
        
        self.logger.log_test_results(metrics, preds, true)
        self.logger.plot_training_curve()
        self.logger.plot_predictions(true, preds)
        
        return metrics, preds, true

if __name__ == '__main__':
    print("训练引擎模块测试")
    print("=" * 60)
    
    y_true = np.array([100, 80, 60, 40, 20, 10, 5])
    y_pred = np.array([95, 75, 65, 35, 25, 8, 7])
    
    metrics = MetricsTracker.compute_all(y_true, y_pred)
    print("指标测试:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
    
    print("\n训练引擎模块测试完成!")
