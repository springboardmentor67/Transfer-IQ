import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

def plot_actual_vs_predicted(y_true, y_pred, model_name, save_path):
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, alpha=0.6, color='blue')
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], color='red', linestyle='--')
    plt.xlabel('Actual Transfer Value')
    plt.ylabel('Predicted Transfer Value')
    plt.title(f'{model_name}: Actual vs Predicted')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f'{model_name}_actual_vs_predicted.png'))
    plt.close()

def plot_model_comparison(results_df, metric, save_path):
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Model', y=metric, data=results_df, palette='viridis')
    plt.title(f'Model Comparison: {metric}')
    plt.ylabel(metric)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f'model_comparison_{metric}.png'))
    plt.close()

def plot_feature_importance(model, feature_names, save_path, top_n=10):
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    else:
        print("Model does not have feature_importances_ attribute.")
        return
        
    indices = np.argsort(importances)[::-1][:top_n]
    
    plt.figure(figsize=(10, 6))
    plt.title("Feature Importances")
    plt.bar(range(top_n), importances[indices], align="center", color='green')
    plt.xticks(range(top_n), [feature_names[i] for i in indices], rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'feature_importance.png'))
    plt.close()

def plot_performance_trends(df_seq, player_id, save_path):
    # Performance over match sequence
    player_data = df_seq[df_seq['player_id'] == player_id].sort_values('match_num')
    
    plt.figure(figsize=(10, 5))
    plt.plot(player_data['match_num'], player_data['goals'], marker='o', label='Goals')
    plt.plot(player_data['match_num'], player_data['assists'], marker='s', label='Assists')
    plt.xlabel('Match Number')
    plt.ylabel('Count')
    plt.title(f'Performance Trend for Player {player_id}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f'player_{player_id}_trends.png'))
    plt.close()
