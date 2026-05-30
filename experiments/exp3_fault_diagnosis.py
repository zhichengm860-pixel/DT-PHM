
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import json
import os

torch.backends.cudnn.enabled = False

from data_loader import CMAPSSLoader, create_dataloaders
from models import LSTMModel, BiLSTMModel, CNNLSTMModel, TransformerModel
from train_engine import Trainer

def create_fault_labels(y_rul, thresholds=None):
    if thresholds is None:
        thresholds = [100, 50, 20]
    
    labels = np.zeros(len(y_rul), dtype=int)
    labels[y_rul <= thresholds[0]] = 1
    labels[y_rul <= thresholds[1]] = 2
    labels[y_rul <= thresholds[2]] = 3
    
    return labels

class FocalLoss(nn.Module):
    
    def __init__(self, num_classes, gamma=2.0, alpha=None, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.num_classes = num_classes
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss

class FaultClassifier(nn.Module):
    
    def __init__(self, input_dim, hidden_dim=64, num_classes=4, 
                 backbone='lstm', num_layers=2, dropout=0.2):
        super().__init__()
        self.backbone_name = backbone
        self.num_classes = num_classes
        
        if backbone == 'lstm':
            self.backbone = nn.LSTM(input_dim, hidden_dim, num_layers, 
                                   batch_first=True, dropout=dropout)
        elif backbone == 'bilstm':
            self.backbone = nn.LSTM(input_dim, hidden_dim, num_layers,
                                   batch_first=True, dropout=dropout, bidirectional=True)
            hidden_dim *= 2
        elif backbone == 'cnn_lstm':
            self.cnn = nn.Sequential(
                nn.Conv1d(input_dim, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool1d(2),
                nn.Conv1d(32, 64, kernel_size=3, padding=1),
                nn.ReLU(),
            )
            self.backbone = nn.LSTM(64, hidden_dim, num_layers,
                                   batch_first=True, dropout=dropout)
        elif backbone == 'transformer':
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=input_dim, nhead=4, dim_feedforward=hidden_dim*2,
                dropout=dropout, batch_first=True
            )
            self.backbone = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            hidden_dim = input_dim
        
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )
        self.fc = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, return_attention=False):
        if self.backbone_name == 'cnn_lstm':
            x = x.permute(0, 2, 1)
            x = self.cnn(x)
            x = x.permute(0, 2, 1)
            lstm_out, _ = self.backbone(x)
        elif self.backbone_name == 'transformer':
            lstm_out = self.backbone(x)
        else:
            lstm_out, _ = self.backbone(x)
        
        attn_weights = F.softmax(self.attention(lstm_out), dim=1)
        context = torch.sum(attn_weights * lstm_out, dim=1)
        context = self.dropout(context)
        out = self.fc(context)
        
        if return_attention:
            return out, attn_weights
        return out

def compute_class_weights(labels):
    class_counts = np.bincount(labels)
    total = len(labels)
    weights = total / (len(class_counts) * class_counts)
    return torch.FloatTensor(weights)

def train_classifier_v2(model, train_loader, val_loader, epochs=30, 
                        device='cuda', use_focal=True, gamma=2.0):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)
    
    all_labels = []
    for _, y in train_loader:
        all_labels.extend(y.numpy())
    class_weights = compute_class_weights(np.array(all_labels)).to(device)
    
    if use_focal:
        criterion = FocalLoss(num_classes=4, gamma=gamma, alpha=class_weights)
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    best_val_acc = 0
    best_model = None
    patience_counter = 0
    patience = 7
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(out, 1)
            train_correct += (predicted == y).sum().item()
            train_total += y.size(0)
        
        model.eval()
        val_correct = 0
        val_total = 0
        val_loss = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                loss = criterion(out, y)
                val_loss += loss.item()
                _, predicted = torch.max(out, 1)
                val_correct += (predicted == y).sum().item()
                val_total += y.size(0)
        
        train_acc = train_correct / train_total
        val_acc = val_correct / val_total
        
        print(f"Epoch {epoch}: Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}, Loss={val_loss/len(val_loader):.4f}")
        
        scheduler.step(val_loss)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break
    
    if best_model is not None:
        model.load_state_dict(best_model)
    
    return model

def evaluate_classifier_v2(model, test_loader, device='cuda'):
    model.eval()
    all_preds = []
    all_true = []
    all_probs = []
    
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            out = model(x)
            probs = F.softmax(out, dim=1)
            _, predicted = torch.max(out, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_true.extend(y.numpy())
            all_probs.extend(probs.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_true = np.array(all_true)
    all_probs = np.array(all_probs)
    
    accuracy = np.mean(all_preds == all_true)
    
    from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_true, all_preds, average='macro', zero_division=0
    )
    
    cm = confusion_matrix(all_true, all_preds)
    
    per_class_precision, per_class_recall, per_class_f1, _ = precision_recall_fscore_support(
        all_true, all_preds, average=None, zero_division=0
    )
    
    return {
        'accuracy': accuracy,
        'precision_macro': precision,
        'recall_macro': recall,
        'f1_macro': f1,
        'confusion_matrix': cm.tolist(),
        'per_class_precision': per_class_precision.tolist(),
        'per_class_recall': per_class_recall.tolist(),
        'per_class_f1': per_class_f1.tolist(),
        'predictions': all_preds.tolist(),
        'true_labels': all_true.tolist(),
        'probabilities': all_probs.tolist()
    }

def run_fault_diagnosis_experiment_v2(device='cuda'):
    print("\n" + "="*80)
    print("实验三v2：故障诊断 (优化版) (C-MAPSS FD001)")
    print("="*80)
    
    loader = CMAPSSLoader(data_dir='./C_MAPSS')
    train_df, test_df, rul_values = loader.load_dataset('FD001')
    
    train_df = loader.add_rul_to_train(train_df)
    test_df = loader.add_rul_to_test(test_df, rul_values)
    
    train_df, test_df, _ = loader.normalize(train_df, test_df)
    
    train_X, train_y, _ = loader.prepare_sequence_data(train_df, seq_length=30)
    test_X, test_y, _ = loader.prepare_sequence_data(test_df, seq_length=30)
    
    train_labels = create_fault_labels(train_y)
    test_labels = create_fault_labels(test_y)
    
    print(f"训练样本: {len(train_X)}, 测试样本: {len(test_X)}")
    print(f"训练标签分布: {np.bincount(train_labels)}")
    print(f"测试标签分布: {np.bincount(test_labels)}")
    
    train_dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(train_X), torch.LongTensor(train_labels)
    )
    test_dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(test_X), torch.LongTensor(test_labels)
    )
    
    train_size = int(0.9 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_subset, val_subset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size]
    )
    
    train_loader = torch.utils.data.DataLoader(train_subset, batch_size=64, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_subset, batch_size=64, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    input_dim = train_X.shape[2]
    results = {}
    
    models_config = [
        ('LSTM+CE', 'lstm', False, 0),
        ('LSTM+Focal', 'lstm', True, 2.0),
        ('BiLSTM+Focal', 'bilstm', True, 2.0),
        ('CNN-LSTM+Focal', 'cnn_lstm', True, 2.0),
        ('Transformer+Focal', 'transformer', True, 2.0),
    ]
    
    for name, backbone, use_focal, gamma in models_config:
        print(f"\n{'='*60}")
        print(f"模型: {name}")
        print('='*60)
        
        model = FaultClassifier(
            input_dim=input_dim, hidden_dim=64, num_classes=4,
            backbone=backbone, num_layers=2
        )
        
        model = train_classifier_v2(
            model, train_loader, val_loader, epochs=30,
            device=device, use_focal=use_focal, gamma=gamma
        )
        
        metrics = evaluate_classifier_v2(model, test_loader, device)
        results[name] = metrics
        
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision_macro']:.4f}")
        print(f"  Recall: {metrics['recall_macro']:.4f}")
        print(f"  F1: {metrics['f1_macro']:.4f}")
        print(f"  混淆矩阵:\n{np.array(metrics['confusion_matrix'])}")
    
    summary = {}
    for name, metrics in results.items():
        summary[name] = {
            'accuracy': metrics['accuracy'],
            'precision': metrics['precision_macro'],
            'recall': metrics['recall_macro'],
            'f1': metrics['f1_macro']
        }
    
    df = pd.DataFrame(summary).T
    print("\n" + "="*80)
    print("结果汇总: exp3_fault_diagnosis_v2")
    print("="*80)
    print(df)
    
    os.makedirs('./experiments', exist_ok=True)
    with open('./experiments/exp3_v2_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    return results

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    
    results = run_fault_diagnosis_experiment_v2(device)
    
    print("\n" + "="*80)
    print("实验三v2完成!")
    print("="*80)

if __name__ == '__main__':
    main()
