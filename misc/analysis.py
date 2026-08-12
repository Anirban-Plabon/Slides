import glob
import json
import os

# Base directory relative to this script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
json_file_path = os.path.join(BASE_DIR, 'history_graph_se_resunet_fusion_v2.json')

with open(json_file_path, 'r') as f:
    history_graph_se_resunet_fusion_v2 = json.load(f)

# Alias for convenience
history = history_graph_se_resunet_fusion_v2


def get_dice(history):
    """
    Returns the max Dice scores across all epochs individually for each region.
    """
    mean_dice = max(history['val_mean'])
    tc_dice = max(history['val_tc'])
    wt_dice = max(history['val_wt'])
    et_dice = max(history['val_et'])
    return mean_dice, tc_dice, wt_dice, et_dice


def get_score(history):
    """
    Returns the Dice scores at the best epoch (where val_mean is maximized).
    """
    best_idx = history['val_mean'].index(max(history['val_mean']))
    mean_dice = history['val_mean'][best_idx]
    tc_dice = history['val_tc'][best_idx]
    wt_dice = history['val_wt'][best_idx]
    et_dice = history['val_et'][best_idx]
    return mean_dice, tc_dice, wt_dice, et_dice


def get_loss(history):
    """
    Returns the loss metrics (train_loss, train_dice, train_focal, train_iou)
    at the best epoch (where val_mean is maximized).
    """
    best_idx = history['val_mean'].index(max(history['val_mean']))
    loss = history['train_loss'][best_idx]
    dice_loss = history['train_dice'][best_idx]
    focal_loss = history['train_focal'][best_idx]
    iou = history['train_iou'][best_idx]
    return loss, dice_loss, focal_loss, iou


def analyze_all_histories(folder=BASE_DIR):
    """
    Analyzes all JSON history files in the folder and returns structured metrics.
    """
    json_files = sorted(glob.glob(os.path.join(folder, '*.json')))
    results = []

    for fpath in json_files:
        model_name = os.path.basename(fpath).replace('_history.json', '').replace('.json', '')
        with open(fpath, 'r') as f:
            h = json.load(f)
        
        best_epoch = h['val_mean'].index(max(h['val_mean'])) + 1
        mean_d, tc, wt, et = get_score(h)
        tr_loss, tr_dice, tr_focal, tr_iou = get_loss(h)
        min_loss = min(h['train_loss'])
        final_loss = h['train_loss'][-1]

        results.append({
            'model': model_name,
            'best_epoch': best_epoch,
            'mean_dice': mean_d,
            'tc_dice': tc,
            'wt_dice': wt,
            'et_dice': et,
            'train_loss': tr_loss,
            'train_dice': tr_dice,
            'train_focal': tr_focal,
            'train_iou': tr_iou,
            'min_loss': min_loss,
            'final_loss': final_loss
        })
    
    return results


if __name__ == '__main__':
    print("==========================================================================")
    print("  Single Model Analysis: Graph SE-ResUNet Fusion v2")
    print("==========================================================================")
    print("Individual Peak Dice (Mean, TC, WT, ET):", get_dice(history))
    print("Best Epoch Dice      (Mean, TC, WT, ET):", get_score(history))
    print("Best Epoch Loss (Total, Dice, Focal, IoU):", get_loss(history))
    
    all_results = analyze_all_histories()

    print("\n" + "=" * 105)
    print("  1. COMPARATIVE VALIDATION DICE SCORES")
    print("=" * 105)
    header_dice = f"{'Model':<38} | {'Best Ep':<7} | {'Mean Dice':<10} | {'TC Dice':<10} | {'WT Dice':<10} | {'ET Dice':<10}"
    print(header_dice)
    print("-" * len(header_dice))
    for r in all_results:
        print(f"{r['model']:<38} | {r['best_epoch']:<7d} | {r['mean_dice']:<10.4f} | {r['tc_dice']:<10.4f} | {r['wt_dice']:<10.4f} | {r['et_dice']:<10.4f}")

    print("\n" + "=" * 115)
    print("  2. COMPARATIVE TRAINING LOSS & LOSS COMPONENTS")
    print("=" * 115)
    header_loss = f"{'Model':<38} | {'Best Ep':<7} | {'Tr Loss':<9} | {'Dice Loss':<9} | {'Focal Loss':<10} | {'Tr IoU':<9} | {'Min Loss':<9} | {'Final Loss':<10}"
    print(header_loss)
    print("-" * len(header_loss))
    for r in all_results:
        print(f"{r['model']:<38} | {r['best_epoch']:<7d} | {r['train_loss']:<9.4f} | {r['train_dice']:<9.4f} | {r['train_focal']:<10.4f} | {r['train_iou']:<9.4f} | {r['min_loss']:<9.4f} | {r['final_loss']:<10.4f}")