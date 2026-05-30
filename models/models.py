
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class LSTMModel(nn.Module):
    
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )
    
    def forward(self, x):
        lstm_out, (h_n, c_n) = self.lstm(x)
        out = self.fc(lstm_out[:, -1, :])
        return out.squeeze(-1)

class BiLSTMModel(nn.Module):
    
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, dropout=0.2):
        super(BiLSTMModel, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = self.fc(lstm_out[:, -1, :])
        return out.squeeze(-1)

class GRUModel(nn.Module):
    
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, dropout=0.2):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )
    
    def forward(self, x):
        gru_out, h_n = self.gru(x)
        out = self.fc(gru_out[:, -1, :])
        return out.squeeze(-1)

class CNNLSTMModel(nn.Module):
    
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, dropout=0.2):
        super(CNNLSTMModel, self).__init__()
        
        self.conv_layers = nn.Sequential(
            nn.Conv1d(input_dim, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.ReLU()
        )
        
        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )
    
    def forward(self, x):
        x = x.permute(0, 2, 1)
        conv_out = self.conv_layers(x)
        conv_out = conv_out.permute(0, 2, 1)
        
        lstm_out, _ = self.lstm(conv_out)
        out = self.fc(lstm_out[:, -1, :])
        return out.squeeze(-1)

class TransformerModel(nn.Module):
    
    def __init__(self, input_dim, d_model=64, nhead=4, num_layers=2, dropout=0.2):
        super(TransformerModel, self).__init__()
        
        self.input_proj = nn.Linear(input_dim, d_model)
        
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.fc = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )
    
    def forward(self, x):
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        x = self.transformer(x)
        x = x.mean(dim=1)
        out = self.fc(x)
        return out.squeeze(-1)

class PositionalEncoding(nn.Module):
    
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class PhysicsModel(nn.Module):
    
    def __init__(self, t_dim=10, hidden_dim=32):
        super(PhysicsModel, self).__init__()
        
        self.model = nn.Sequential(
            nn.Linear(t_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, t_degradation):
        if t_degradation.dim() == 3:
            t_degradation = t_degradation[:, -1, :]
        return self.model(t_degradation).squeeze(-1)
    
    def physics_constraint(self, rul_pred, t_degradation):
        constraint_loss = 0.0
        
        degradation_severity = torch.abs(t_degradation - 1.0).mean(dim=-1)
        
        expected_rul = 100 * (1 - degradation_severity)
        monotonicity_loss = F.relu(rul_pred - expected_rul - 20).mean()
        constraint_loss += monotonicity_loss
        
        boundary_loss = F.relu(-rul_pred).mean() + F.relu(rul_pred - 150).mean()
        constraint_loss += boundary_loss
        
        return constraint_loss

class DTPHMCollaborativeModel(nn.Module):
    
    def __init__(self, input_dim, t_dim=10, w_dim=4, 
                 hidden_dim=64, num_layers=2, dropout=0.2,
                 use_physics=True, use_constraint=True):
        super(DTPHMCollaborativeModel, self).__init__()
        
        self.use_physics = use_physics
        self.use_constraint = use_constraint
        self.t_dim = t_dim
        self.w_dim = w_dim
        
        if use_physics:
            self.physics_model = PhysicsModel(t_dim=t_dim, hidden_dim=32)
        
        self.data_encoder = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        self.condition_attention = nn.Sequential(
            nn.Linear(w_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
        fusion_dim = hidden_dim * 2
        if use_physics:
            fusion_dim += 1
        
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
        if use_physics:
            self.residual_head = nn.Sequential(
                nn.Linear(hidden_dim * 2, 32),
                nn.ReLU(),
                nn.Linear(32, 1)
            )
    
    def forward(self, x, t_degradation=None, w_condition=None):
        batch_size = x.size(0)
        
        lstm_out, _ = self.data_encoder(x)
        
        if w_condition is not None and w_condition.dim() == 3:
            attn_weights = self.condition_attention(w_condition)
            attn_weights = F.softmax(attn_weights, dim=1)
            data_features = (lstm_out * attn_weights).sum(dim=1)
        else:
            data_features = lstm_out[:, -1, :]
        
        if self.use_physics and t_degradation is not None:
            physics_pred = self.physics_model(t_degradation)
            
            residual = self.residual_head(data_features).squeeze(-1)
            
            collaborative_pred = physics_pred + residual
            
            fusion_input = torch.cat([data_features, physics_pred.unsqueeze(-1)], dim=-1)
            fused_pred = self.fusion(fusion_input).squeeze(-1)
            
            alpha = 0.5
            rul_pred = alpha * collaborative_pred + (1 - alpha) * fused_pred
            
            return rul_pred, physics_pred, residual
        else:
            rul_pred = self.fusion(data_features).squeeze(-1)
            return rul_pred
    
    def compute_physics_loss(self, rul_pred, t_degradation):
        if not self.use_constraint or t_degradation is None:
            return torch.tensor(0.0, device=rul_pred.device)
        
        return self.physics_model.physics_constraint(rul_pred, t_degradation)
    
    def update_physics_model(self, t_degradation, rul_true, lr=0.001):
        if not self.use_physics:
            return
        
        self.physics_model.train()
        optimizer = torch.optim.Adam(self.physics_model.parameters(), lr=lr)
        
        physics_pred = self.physics_model(t_degradation)
        loss = F.mse_loss(physics_pred, rul_true)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

class FaultDiagnosisModel(nn.Module):
    
    def __init__(self, input_dim, num_classes=4, hidden_dim=64, num_layers=2, dropout=0.2):
        super(FaultDiagnosisModel, self).__init__()
        
        self.encoder = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )
    
    def forward(self, x):
        lstm_out, _ = self.encoder(x)
        features = lstm_out[:, -1, :]
        logits = self.classifier(features)
        return logits

def get_model(model_name, input_dim, **kwargs):
    models = {
        'lstm': LSTMModel,
        'bilstm': BiLSTMModel,
        'gru': GRUModel,
        'cnn_lstm': CNNLSTMModel,
        'transformer': TransformerModel,
        'dt_phm': DTPHMCollaborativeModel,
        'fault_diagnosis': FaultDiagnosisModel
    }
    
    if model_name.lower() not in models:
        raise ValueError(f"未知模型: {model_name}，可用模型: {list(models.keys())}")
    
    return models[model_name.lower()](input_dim, **kwargs)

if __name__ == '__main__':
    print("=" * 60)
    print("模型结构测试")
    print("=" * 60)
    
    batch_size = 4
    seq_len = 30
    input_dim = 17
    t_dim = 10
    w_dim = 4
    
    x = torch.randn(batch_size, seq_len, input_dim)
    t = torch.randn(batch_size, t_dim)
    w = torch.randn(batch_size, seq_len, w_dim)
    
    models_to_test = [
        ('LSTM', LSTMModel(input_dim)),
        ('BiLSTM', BiLSTMModel(input_dim)),
        ('GRU', GRUModel(input_dim)),
        ('CNN-LSTM', CNNLSTMModel(input_dim)),
        ('Transformer', TransformerModel(input_dim)),
    ]
    
    for name, model in models_to_test:
        output = model(x)
        print(f"{name}: 输入{x.shape} -> 输出{output.shape}")
    
    print("\n" + "=" * 60)
    print("DT-PHM协同模型测试")
    print("=" * 60)
    
    dt_model = DTPHMCollaborativeModel(
        input_dim=input_dim, 
        t_dim=t_dim, 
        w_dim=w_dim,
        use_physics=True, 
        use_constraint=True
    )
    
    rul_pred, physics_pred, residual = dt_model(x, t, w)
    print(f"DT-PHM模型输出:")
    print(f"  RUL预测: {rul_pred.shape}")
    print(f"  物理预测: {physics_pred.shape}")
    print(f"  残差: {residual.shape}")
    
    physics_loss = dt_model.compute_physics_loss(rul_pred, t)
    print(f"  物理约束损失: {physics_loss.item():.4f}")
    
    fault_model = FaultDiagnosisModel(input_dim, num_classes=4)
    fault_logits = fault_model(x)
    print(f"\n故障诊断模型: 输入{x.shape} -> 输出{fault_logits.shape}")
    
    print("\n模型测试完成!")
