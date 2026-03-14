import os
import numpy as np
from scipy.stats import kruskal, mannwhitneyu
import matplotlib.pyplot as plt

RESULTS_DIR = "results2"

MODELS = ["ACE", "ATT", "LAT"]

def model_from_filename(fname):
    for m in MODELS:
        if fname.startswith(m):
            return m
    return None


# Containers
recall = {m: [] for m in MODELS}
f1 = {m: [] for m in MODELS}

tpr = {m: [] for m in MODELS}
tnr = {m: [] for m in MODELS}
fpr = {m: [] for m in MODELS}
fnr = {m: [] for m in MODELS}


def compute_f1(tp, fp, fn):
    denom = 2 * tp + fp + fn
    if denom == 0:
        return 0.0
    return (2 * tp) / denom


# Load data
for fname in os.listdir(RESULTS_DIR):
    if not fname.endswith(".npz"):
        continue

    model = model_from_filename(fname)
    if model is None:
        continue

    data = np.load(os.path.join(RESULTS_DIR, fname))

    # last epoch only
    tn = data["tn"][-1]
    fp = data["fp"][-1]
    fn = data["fn"][-1]
    tp = data["tp"][-1]

    # Recall
    recall_val = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    recall[model].append(recall_val)

    # F1 (computed)
    f1[model].append(compute_f1(tp, fp, fn))

    # Confusion matrix percentages
    tpr[model].append(recall_val)
    tnr[model].append(tn / (tn + fp) if (tn + fp) > 0 else 0.0)
    fpr[model].append(fp / (fp + tn) if (fp + tn) > 0 else 0.0)
    fnr[model].append(fn / (fn + tp) if (fn + tp) > 0 else 0.0)


# -------------------------
# Statistical tests
# -------------------------

def kruskal_test(metric_dict, name):
    samples = [metric_dict[m] for m in MODELS]
    stat, p = kruskal(*samples)
    print(f"\n{name}")
    print(f"Kruskal–Wallis H={stat:.4f}, p={p:.4e}")
    return p


def print_metric_summary(metric_dict, name):
    print(f"\nSummary statistics for {name}:")
    for model in MODELS:
        values = np.array(metric_dict[model])
        print(f"  {model}: mean={values.mean():.4f}, std={values.std():.4f}, min={values.min():.4f}, max={values.max():.4f}")


def pairwise_tests(metric_dict, name):
    print(f"\nPairwise Mann–Whitney U for {name}")
    for i in range(len(MODELS)):
        for j in range(i + 1, len(MODELS)):
            m1, m2 = MODELS[i], MODELS[j]
            stat, p = mannwhitneyu(
                metric_dict[m1],
                metric_dict[m2],
                alternative="two-sided"
            )
            print(f"{m1} vs {m2}: U={stat:.2f}, p={p:.4e}")


def plot_pairwise_tests(metric_dict, name):
    """Create a visualization of pairwise Mann-Whitney U test results"""
    pairwise_results = []
    pairwise_labels = []
    
    for i in range(len(MODELS)):
        for j in range(i + 1, len(MODELS)):
            m1, m2 = MODELS[i], MODELS[j]
            stat, p = mannwhitneyu(
                metric_dict[m1],
                metric_dict[m2],
                alternative="two-sided"
            )
            pairwise_results.append(p)
            pairwise_labels.append(f"{m1} vs {m2}")
    
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ['#2ca02c' if p < 0.05 else '#d62728' for p in pairwise_results]
    
    bars = ax.bar(pairwise_labels, pairwise_results, color=colors, alpha=0.7)
    ax.axhline(y=0.05, color='black', linestyle='--', linewidth=2, label='Significance threshold (p=0.05)')
    ax.set_ylabel('p-value')
    ax.set_title(f'Pairwise Mann–Whitney U Tests for {name}')
    ax.set_ylim(0, max(pairwise_results) * 1.1)
    ax.grid(axis='y', alpha=0.3)
    ax.legend()
    
    # Add value labels on bars
    for bar, p in zip(bars, pairwise_results):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height, f'{p:.4e}', 
                ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    filename = f"pairwise_{name.replace(' ', '_').replace('(', '').replace(')', '')}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Saved figure: {filename}")
    plt.close()


def plot_metric(metric_dict, name):
    """Create visualizations for a metric: bar plot with error bars and box plot"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Bar plot with error bars
    means = [np.mean(metric_dict[m]) for m in MODELS]
    stds = [np.std(metric_dict[m]) for m in MODELS]
    
    ax1.bar(MODELS, means, yerr=stds, capsize=5, alpha=0.7, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    ax1.set_ylabel('Value')
    ax1.set_title(f'{name} - Mean ± Std Dev')
    ax1.set_ylim(0, 1.0)
    ax1.grid(axis='y', alpha=0.3)
    
    # Box plot
    data_for_box = [metric_dict[m] for m in MODELS]
    bp = ax2.boxplot(data_for_box, labels=MODELS, patch_artist=True)
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax2.set_ylabel('Value')
    ax2.set_title(f'{name} - Distribution')
    ax2.set_ylim(0, 1.0)
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    filename = f"{name.replace(' ', '_').replace('(', '').replace(')', '')}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Saved figure: {filename}")
    plt.close()


def plot_statistical_summary(test_results):
    """Create a summary plot of p-values from Kruskal-Wallis tests"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    names = [result['name'] for result in test_results]
    p_values = [result['p_value'] for result in test_results]
    colors = ['#2ca02c' if p < 0.05 else '#d62728' for p in p_values]
    
    bars = ax.barh(names, p_values, color=colors, alpha=0.7)
    ax.axvline(x=0.05, color='black', linestyle='--', linewidth=2, label='Significance threshold (p=0.05)')
    ax.set_xlabel('p-value')
    ax.set_title('Kruskal-Wallis Test Results (Statistical Significance)')
    ax.set_xlim(0, max(p_values) * 1.1)
    ax.grid(axis='x', alpha=0.3)
    ax.legend()
    
    # Add value labels on bars
    for bar, p in zip(bars, p_values):
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height()/2, f'{p:.4e}', 
                ha='left', va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('statistical_tests_summary.png', dpi=300, bbox_inches='tight')
    print(f"Saved figure: statistical_tests_summary.png")
    plt.close()


# Store test results for summary
test_results = []

# Recall
p = kruskal_test(recall, "Recall")
test_results.append({'name': 'Recall', 'p_value': p})
plot_metric(recall, "Recall")
if p < 0.05:
    print_metric_summary(recall, "Recall")
    pairwise_tests(recall, "Recall")

# F1
p = kruskal_test(f1, "F1 score (computed)")
test_results.append({'name': 'F1 score (computed)', 'p_value': p})
if p < 0.05:
    print_metric_summary(f1, "F1 score (computed)")
    pairwise_tests(f1, "F1 score (computed)")
    plot_metric(f1, "F1 score (computed)")
    plot_pairwise_tests(f1, "F1 score (computed)")

# Confusion matrix percentages
for metric, name in [
    (tpr, "TPR (Recall)"),
    (tnr, "TNR"),
    (fpr, "FPR"),
    (fnr, "FNR"),
]:
    p = kruskal_test(metric, name)
    test_results.append({'name': name, 'p_value': p})
    if p < 0.05:
        print_metric_summary(metric, name)
        pairwise_tests(metric, name)
        plot_metric(metric, name)
        # Create pairwise visualization for FPR
        if name == "FPR":
            plot_pairwise_tests(metric, name)

# Create summary plot of all statistical tests
plot_statistical_summary(test_results)
