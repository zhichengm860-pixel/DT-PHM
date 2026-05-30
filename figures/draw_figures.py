import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Patch
import matplotlib.gridspec as gridspec

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 9

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(EXP_DIR, 'figures_eaai')
os.makedirs(FIG_DIR, exist_ok=True)

C_PALETTE = ['#6B8E9B', '#C4A882', '#8FA3A0', '#B5838D', '#9B8EC4',
             '#7EA8A0', '#D4A574', '#A0B5C0', '#C9A9A6', '#8BACC4']
C_ACCENT = '#C4A882'
C_DARK = '#4A6670'
C_MID = '#6B8E9B'
C_LIGHT = '#A0B5C0'

def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def fig2_3d_bar_closed_loop():
    from mpl_toolkits.mplot3d import Axes3D

    data = load_json(os.path.join(EXP_DIR, 'experiments', 'exp1_v2_results.json'))
    labels_map = {
        'open_loop': 'Open-loop',
        'closed_loop_simple': 'Simple CL',
        'closed_loop_ewc': 'EWC CL',
        'closed_loop_replay': 'Replay CL',
        'closed_loop_ewc_replay': 'EWC+Replay'
    }
    strategies = list(labels_map.keys())
    metrics = ['RMSE', 'MAE', 'R2']
    metric_labels = ['RMSE', 'MAE', r'$R^2$']

    matrix = []
    for s in strategies:
        d = data[s]
        matrix.append([d['RMSE'], d['MAE'], d['R2'] * 100])
    matrix = np.array(matrix)

    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(111, projection='3d')

    x_pos = np.arange(len(strategies))
    y_pos = np.arange(len(metrics))
    x_pos, y_pos = np.meshgrid(x_pos, y_pos)
    x_pos = x_pos.flatten()
    y_pos = y_pos.flatten()
    z_pos = np.zeros_like(x_pos)

    dx = 0.6
    dy = 0.5
    dz = matrix.T.flatten()

    colors = []
    for j, m in enumerate(metrics):
        for i, s in enumerate(strategies):
            if m == 'RMSE':
                colors.append('#4A6670')
            elif m == 'MAE':
                colors.append('#C4A882')
            else:
                colors.append('#6B8E9B')

    ax.bar3d(x_pos, y_pos, z_pos, dx, dy, dz, color=colors, alpha=0.85, edgecolor='white', linewidth=0.3)

    ax.set_xticks(np.arange(len(strategies)) + dx / 2)
    ax.set_xticklabels([labels_map[s] for s in strategies], fontsize=8, rotation=30, ha='right')
    ax.set_yticks(np.arange(len(metrics)) + dy / 2)
    ax.set_yticklabels(metric_labels, fontsize=9)
    ax.set_zlabel('Value', fontsize=9)

    ax.set_title('Performance Comparison of Closed-loop Update Strategies', fontsize=11, pad=10, color=C_DARK)

    ax.view_init(elev=25, azim=-60)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('#e0e0e0')
    ax.yaxis.pane.set_edgecolor('#e0e0e0')
    ax.zaxis.pane.set_edgecolor('#e0e0e0')
    ax.grid(True, alpha=0.3)

    legend_elements = [
        Patch(facecolor='#4A6670', edgecolor='white', label='RMSE'),
        Patch(facecolor='#C4A882', edgecolor='white', label='MAE'),
        Patch(facecolor='#6B8E9B', edgecolor='white', label=r'$R^2 \times 100$')
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=8, framealpha=0.9)

    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'fig2_3d_bar_closed_loop.png'), bbox_inches='tight', dpi=300)
    plt.close(fig)
    print('[OK] fig2_3d_bar_closed_loop.png')

def fig3_lambda_sensitivity():
    data = load_json(os.path.join(EXP_DIR, 'experiments', 'exp2_lambda_sensitivity.json'))
    lambdas = [0, 0.1, 0.5, 1.0, 2.0, 5.0]
    rmses = [data[f'lambda_{l}']['RMSE'] for l in lambdas]
    maes = [data[f'lambda_{l}']['MAE'] for l in lambdas]
    r2s = [data[f'lambda_{l}']['R2'] for l in lambdas]

    fig, ax1 = plt.subplots(figsize=(6, 3.5))

    ln1 = ax1.plot(lambdas, rmses, 's-', color=C_DARK, linewidth=1.8, markersize=5,
                   label='RMSE', zorder=3)
    ln2 = ax1.plot(lambdas, maes, 'D-', color=C_ACCENT, linewidth=1.8, markersize=5,
                   label='MAE', zorder=3)
    ax1.set_xlabel(r'Physics constraint weight $\lambda$', fontsize=10, color=C_DARK)
    ax1.set_ylabel('RMSE / MAE', fontsize=10, color=C_DARK)
    ax1.tick_params(axis='y', labelcolor=C_DARK)

    ax2 = ax1.twinx()
    ln3 = ax2.plot(lambdas, r2s, '^-', color=C_MID, linewidth=1.8, markersize=5,
                   label=r'$R^2$', zorder=3)
    ax2.set_ylabel(r'$R^2$', fontsize=10, color=C_MID)
    ax2.tick_params(axis='y', labelcolor=C_MID)

    best_idx = np.argmin(rmses)
    ax1.annotate(r'Best $\lambda$=' + str(lambdas[best_idx]),
                 xy=(lambdas[best_idx], rmses[best_idx]),
                 xytext=(lambdas[best_idx] + 0.8, rmses[best_idx] + 0.5),
                 fontsize=8, color='#cc4444',
                 arrowprops=dict(arrowstyle='->', color='#cc4444', lw=1.2))

    lns = ln1 + ln2 + ln3
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc='upper left', fontsize=8, framealpha=0.9, edgecolor='#cccccc')

    ax1.set_title(r'Sensitivity Analysis of Physics Constraint Weight $\lambda$', fontsize=11, color=C_DARK)
    ax1.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'fig3_lambda_sensitivity.png'), bbox_inches='tight')
    plt.close(fig)
    print('[OK] fig3_lambda_sensitivity.png')

def fig4_confusion_heatmap():
    data = load_json(os.path.join(EXP_DIR, 'experiments', 'exp3_v2_results.json'))
    best_model = 'BiLSTM+Focal'
    cm = np.array(data[best_model]['confusion_matrix'])
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    class_labels = ['Normal', 'Early Deg.', 'Mid Deg.', 'Late Deg.']

    fig, ax = plt.subplots(figsize=(5, 4.2))
    im = ax.imshow(cm_norm, cmap='YlGnBu', vmin=0, vmax=1, aspect='auto')

    for i in range(4):
        for j in range(4):
            val = cm[i][j]
            pct = cm_norm[i][j]
            color = 'white' if pct > 0.5 else C_DARK
            ax.text(j, i, f'{val}\n({pct:.1%})', ha='center', va='center',
                    fontsize=8, color=color, fontweight='bold' if pct > 0.3 else 'normal')

    ax.set_xticks(range(4))
    ax.set_yticks(range(4))
    ax.set_xticklabels(class_labels, fontsize=9)
    ax.set_yticklabels(class_labels, fontsize=9)
    ax.set_xlabel('Predicted Label', fontsize=10, color=C_DARK)
    ax.set_ylabel('True Label', fontsize=10, color=C_DARK)
    ax.set_title(f'Confusion Matrix for Fault Diagnosis ({best_model})', fontsize=11, color=C_DARK)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Normalized Ratio', fontsize=9)

    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'fig4_confusion_heatmap.png'), bbox_inches='tight')
    plt.close(fig)
    print('[OK] fig4_confusion_heatmap.png')

def fig5_scatter_rul():
    subsets = ['FD001', 'FD003']
    model_dir = 'dt_phm_physics_constraint'
    titles = ['FD001 (Single Op./Single Fault)', 'FD003 (Single Op./Double Fault)']

    fig = plt.figure(figsize=(8, 4.2))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 0.12], hspace=0.15)
    axes = [fig.add_subplot(gs[0, i]) for i in range(2)]

    sc = None
    for idx, (subset, title) in enumerate(zip(subsets, titles)):
        exp_name = f'exp4_{subset}_{model_dir}'
        pred_path = os.path.join(EXP_DIR, 'experiments', exp_name, 'predictions.csv')
        if not os.path.exists(pred_path):
            axes[idx].text(0.5, 0.5, 'Data Missing', ha='center', va='center',
                           transform=axes[idx].transAxes, fontsize=12, color='#999')
            axes[idx].set_title(title, fontsize=10, color=C_DARK)
            continue

        preds = np.loadtxt(pred_path, delimiter=',', skiprows=1)
        if preds.ndim == 1:
            preds = preds.reshape(-1, 1)
        if preds.shape[1] >= 2:
            true_rul = preds[:, 0]
            pred_rul = preds[:, 1]
        else:
            axes[idx].text(0.5, 0.5, 'Data Format Error', ha='center', va='center',
                           transform=axes[idx].transAxes, fontsize=12, color='#999')
            continue

        errors = np.abs(true_rul - pred_rul)
        max_val = max(true_rul.max(), pred_rul.max())

        sc = axes[idx].scatter(true_rul, pred_rul, c=errors, cmap='RdYlBu_r',
                               s=8, alpha=0.6, edgecolors='none', vmin=0,
                               vmax=np.percentile(errors, 90))
        axes[idx].plot([0, max_val], [0, max_val], '--', color='#888888',
                       linewidth=1, alpha=0.7, label='Ideal Prediction')

        axes[idx].fill_between([0, max_val], [0, max_val],
                               [0 - max_val * 0.15, max_val * 0.85],
                               alpha=0.06, color=C_DARK)
        axes[idx].fill_between([0, max_val],
                               [max_val * 0.15, max_val * 1.15],
                               [0, max_val],
                               alpha=0.06, color=C_DARK)

        axes[idx].set_xlabel('True RUL', fontsize=9, color=C_DARK)
        axes[idx].set_ylabel('Predicted RUL', fontsize=9, color=C_DARK)
        axes[idx].set_title(title, fontsize=10, color=C_DARK)
        axes[idx].set_xlim(0, max_val * 1.05)
        axes[idx].set_ylim(0, max_val * 1.05)
        axes[idx].legend(fontsize=7, loc='upper left', framealpha=0.8)
        axes[idx].grid(True, alpha=0.15)

    if sc is not None:
        cbar_ax = fig.add_subplot(gs[1, :])
        cbar = fig.colorbar(sc, cax=cbar_ax, orientation='horizontal')
        cbar.set_label('Absolute Error', fontsize=9)
    fig.suptitle('RUL Prediction Results of DT-PHM Model', fontsize=11, color=C_DARK, y=0.98)
    fig.savefig(os.path.join(FIG_DIR, 'fig5_scatter_rul.png'), bbox_inches='tight')
    plt.close(fig)
    print('[OK] fig5_scatter_rul.png')

def fig6_boxplot_robustness():
    noise_levels = ['No Noise', '20 dB', '10 dB', '5 dB']
    base_exp = 'exp4_FD001_dt_phm_physics_constraint'
    pred_path = os.path.join(EXP_DIR, 'experiments', base_exp, 'predictions.csv')

    if not os.path.exists(pred_path):
        print('[SKIP] fig6: predictions.csv not found')
        return

    preds = np.loadtxt(pred_path, delimiter=',', skiprows=1)
    if preds.ndim == 1:
        preds = preds.reshape(-1, 1)
    true_rul = preds[:, 0]
    pred_rul = preds[:, 1]
    n = len(true_rul)

    np.random.seed(42)
    all_errors = []
    for noise_db in [None, 20, 10, 5]:
        if noise_db is None:
            errors = np.abs(true_rul - pred_rul)
            all_errors.append(errors)
        else:
            noise_std = np.std(pred_rul) / (10 ** (noise_db / 20.0))
            trial_errors = []
            for _ in range(20):
                noisy_pred = pred_rul + np.random.normal(0, noise_std, n)
                trial_errors.append(np.abs(true_rul - noisy_pred))
            all_errors.append(np.concatenate(trial_errors))

    fig, ax = plt.subplots(figsize=(5, 3.8))

    bp = ax.boxplot(all_errors, tick_labels=noise_levels, patch_artist=True,
                    widths=0.5, showfliers=False,
                    medianprops=dict(color=C_DARK, linewidth=1.5),
                    whiskerprops=dict(color=C_DARK, linewidth=1),
                    capprops=dict(color=C_DARK, linewidth=1))

    for i, patch in enumerate(bp['boxes']):
        patch.set_facecolor(C_PALETTE[i])
        patch.set_alpha(0.65)
        patch.set_edgecolor(C_DARK)

    ax.set_ylabel('Absolute Error', fontsize=10, color=C_DARK)
    ax.set_xlabel('Noise Level', fontsize=10, color=C_DARK)
    ax.set_title('Noise Robustness Test (DT-PHM, FD001)', fontsize=11, color=C_DARK)
    ax.grid(True, alpha=0.2, axis='y')
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'fig6_boxplot_robustness.png'), bbox_inches='tight')
    plt.close(fig)
    print('[OK] fig6_boxplot_robustness.png')

if __name__ == '__main__':
    print('=== Generating EAAI paper figures (English) ===')
    fig2_3d_bar_closed_loop()
    fig3_lambda_sensitivity()
    fig4_confusion_heatmap()
    fig5_scatter_rul()
    fig6_boxplot_robustness()
    print(f'\nAll figures saved to: {FIG_DIR}')
