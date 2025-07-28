import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import re

# Parse results from the output text file
def parse_iptw_results(filename):
    results = []
    subgroup = None
    level = None
    outcome = None
    with open(filename, 'r') as f:
        for line in f:
            # Try to extract subgroup and level from previous lines
            if 'IPTW Causal Effect by' in line:
                subgroup = line.split('by ')[-1].strip().replace('=====', '').strip()
            elif line.strip().startswith('---'):
                # e.g., --- Gender: F ---
                m = re.match(r'--- (.*?): (.*?) ---', line.strip())
                if m:
                    subgroup = m.group(1).strip()
                    level = m.group(2).strip()
            elif 'Estimated causal effect' in line:
                # e.g., Estimated causal effect (ATE) of high adherence (Q5 vs Q1) on BP_CHANGE: -3.58
                m = re.match(r'.*on ([A-Z_]+): ([\-0-9\.]+)', line)
                if m:
                    outcome = m.group(1)
                    ate = float(m.group(2))
                    # CI and p-value not available in this output, so set as NaN
                    results.append({'Subgroup': subgroup, 'Level': level, 'Outcome': outcome, 'ATE': ate, 'CI_L': np.nan, 'CI_U': np.nan, 'p': np.nan})
    return pd.DataFrame(results)

# Load results from file
df = parse_iptw_results('iptw_subgroup_results.txt')

# For plotting, group by outcome and plot all subgroups/levels

def plot_forest(df, outcome, title, filename):
    subdf = df[df['Outcome'] == outcome].copy()
    subdf['label'] = subdf['Subgroup'] + ': ' + subdf['Level']
    subdf = subdf.sort_values('ATE')
    y_pos = np.arange(len(subdf))
    fig, ax = plt.subplots(figsize=(10, max(4, len(subdf)*0.5)))
    # Since CI is not available, plot only point estimates
    ax.errorbar(subdf['ATE'], y_pos, xerr=None, fmt='o', color='navy', ecolor='gray', capsize=4)
    ax.axvline(0, color='red', linestyle='--', lw=1)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(subdf['label'])
    ax.set_xlabel('Causal Effect (ATE, Q5 vs Q1)')
    ax.set_title(title)
    for i, ate in enumerate(subdf['ATE']):
        ax.text(ate + 0.1, y_pos[i], f'{ate:.2f}', va='center', fontsize=10)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.show()

# Forest plots for each outcome
plot_forest(df, 'BP_CHANGE', 'Causal Effect of High Adherence on BP Change (by Subgroup)', 'forest_bp_change.png')
plot_forest(df, 'A1C_CHANGE', 'Causal Effect of High Adherence on A1C Change (by Subgroup)', 'forest_a1c_change.png')
plot_forest(df, 'WEIGHT_CHANGE', 'Causal Effect of High Adherence on Weight Change (by Subgroup)', 'forest_weight_change.png') 