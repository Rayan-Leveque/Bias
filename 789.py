import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import warnings

warnings.filterwarnings('ignore')

# Read data
implicit_df = pd.read_csv('result_implicit.csv', index_col=0)

print("Original data shape:", implicit_df.shape)
print("Datasets in CSV:", sorted(implicit_df['dataset'].unique()))
print("Models in CSV:", implicit_df['model'].unique())
print("-" * 50)

# Set seaborn theme
sns.set_theme(style="whitegrid")

# Replace model names for cleaner display
implicit_df['model'] = implicit_df['model'].replace({
    'Qwen3-30B-A3B-Instruct-2507': 'Qwen3-30B',
    'openai/gpt-oss-20b': 'GPT-OSS-20B',
    'Mistral-Small-3.2-24B': 'Mistral-Small-3.2',
    'Phi-4': 'Phi-4'
})

# Define the order for models (columns)
column_order = ['Qwen3-30B', 'Phi-4', 'GPT-OSS-20B', 'Mistral-Small-3.2']

# Categories that exist in your CSV (using 'dataset' column)
# Note: 'arab/muslim' in CSV, but we'll rename it to 'arab' to match the original code
datasets_in_csv = ['age', 'career', 'power', 'science', 'sexuality', 
                   'racism', 'weapon', 'guilt', 'skintone', 
                   'arab/muslim', 'asian', 'black', 'hispanic', 'english']

# Rename 'arab/muslim' to 'arab' to match expected format
implicit_df['dataset'] = implicit_df['dataset'].replace({'arab/muslim': 'arab'})

# Now define the full category order (from original code)
cat_order = ['racism', 'guilt', 'skintone', 'weapon', 'black', 'hispanic', 'asian', 'arab', 'english',
            'career', 'science', 'power', 'sexuality',
            'islam', 'judaism', 'buddhism',
            'disability', 'weight', 'age', 'mentalill', 'eating']

# Filter to only include categories that exist in your data
cat_order_filtered = [cat for cat in cat_order if cat in implicit_df['dataset'].unique()]

print(f"Categories in filtered order: {cat_order_filtered}")
print(f"Number of categories: {len(cat_order_filtered)}")
print("-" * 50)

# Define colors for each category (from original code)
category_colors = {
    'racism': 'coral',
    'guilt': 'coral',
    'skintone': 'coral',
    'weapon': 'coral',
    'black': 'coral',
    'hispanic': 'coral',
    'asian': 'coral',
    'arab': 'coral',
    'english': 'coral',
    'career': 'black',
    'science': 'black',
    'power': 'black',
    'sexuality': 'black',
    'islam': 'green',
    'judaism': 'green',
    'buddhism': 'green',
    'disability': 'blue',
    'weight': 'blue',
    'age': 'blue',
    'mentalill': 'blue',
    'eating': 'blue'
}

# Filter data to only include categories in our order
implicit_df_filtered = implicit_df[implicit_df['dataset'].isin(cat_order_filtered)]

print(f"Filtered data shape: {implicit_df_filtered.shape}")
print("-" * 50)

# Create the plot
g = sns.catplot(
    data=implicit_df_filtered, 
    x="dataset",
    y="iat_bias", 
    col='model', 
    col_wrap=2,
    capsize=.2, 
    palette=category_colors, 
    hue="dataset",
    legend=False,
    errorbar="ci",
    kind="point", 
    height=5.5, 
    aspect=2.5, 
    order=cat_order_filtered, 
    col_order=column_order,
    sharex=True,
    sharey=True
)

# Set y-axis limits
g.set(ylim=(-1, 1.2))

# Force all subplots to show x-axis labels
g.set_titles("{col_name}", size=26)
for ax in g.axes.flat:
    ax.tick_params(labelbottom=True)

# Customize each subplot
for ax in g.axes.flat:
    # Gray background band
    ax.fill_between(x=[-0.5, len(cat_order_filtered)-0.5], y1=0.05, y2=1.1, 
                     color='gray', alpha=0.08, zorder=0)
    
    # Clean up title
    ax_title = ax.get_title()
    if 'model = ' in ax_title:
        new_title = ax_title.replace('model = ', '')
        ax.set_title(new_title, size=26)
    
    # Add horizontal reference line at y=0
    ax.axhline(0, ls='--', c='red', linewidth=2)
    
    # Set axis labels
    ax.set_ylabel('Implicit Bias', fontsize=26)
    ax.set_xlabel('')
    
    # Customize x-tick labels - FORCE THEM TO SHOW ON ALL SUBPLOTS
    ax.set_xticks(range(len(cat_order_filtered)))
    ax.set_xticklabels(cat_order_filtered, rotation=45, ha='right', fontsize=26)
    ax.tick_params(axis='x', labelbottom=True)  # Force x labels to show
    
    # Color the x-tick labels according to category
    for label in ax.get_xticklabels():
        text = label.get_text()
        if text in category_colors:
            label.set_color(category_colors[text])
    
    # Adjust y-tick label size
    for label in ax.get_yticklabels():
        label.set_size(20)

# Keep spines and show y-axis on all subplots
g.despine(left=False)

# Ensure y-axis labels are shown on all subplots
for ax in g.axes.flat:
    ax.yaxis.set_tick_params(labelleft=True)
    ax.tick_params(axis='y', labelsize=20)

# Tight layout to prevent label cutoff
plt.tight_layout()

# Save figure
output_path = 'implicit_bias_fixed.pdf'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Figure saved to: {output_path}")

# Also save as PNG for easier viewing
output_path_png = 'implicit_bias_fixed.png'
plt.savefig(output_path_png, dpi=300, bbox_inches='tight')
print(f"Figure also saved as PNG to: {output_path_png}")

plt.show()

print("\n" + "=" * 50)
print("SUMMARY STATISTICS")
print("=" * 50)
summary = implicit_df_filtered.groupby(['model', 'dataset'])['iat_bias'].agg(['mean', 'std', 'count'])
print(summary)