import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Function to create a radar chart
def create_radar_chart(data, categories, model_names, ax):
    # Set the number of variables (categories)
    num_vars = len(categories)
    
    # Compute angle for each category
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    
    # Close the loop for angles
    angles += angles[:1]
    
    # Plot each model's data
    for i, model_name in enumerate(model_names):
        # Extract the row for the model and dataset
        model_data = data[data['Model'] == model_name].iloc[0]  
        
        # Extract the values for the categories and convert them to a list
        values = model_data[categories].tolist()
        
        # Close the loop for values
        values += values[:1]
        
        # Check if lengths match
        if len(values) != len(angles):
            raise ValueError(f"Length of values {len(values)} does not match length of angles {len(angles)}.")
        
        # Plot the radar chart
        ax.fill(angles, values, alpha=0.25, label=model_name, color=('b' if i == 0 else 'r'))
        ax.plot(angles, values, linewidth=2, color=('b' if i == 0 else 'r'))

    # Add labels to each axis
    ax.set_yticklabels([])  
    ax.set_xticks(angles[:-1])  
    ax.set_xticklabels(categories, fontsize=22, color='black')
    
    # Title with dataset name
    ax.set_title(f'{data["Dataset"].iloc[0]}', size=24, color='black', weight='bold', va='bottom')
    
    # Add a legend
    ax.legend(loc='lower center', fontsize=22, bbox_to_anchor=(0.5, -0.4), ncol=1, frameon=False)

# Prepare the dataset
data = {
    'Dataset': ['OCTMNIST (5k)', 'OCTMNIST (5k)', 'OCTID (0.5k)', 'OCTID (0.5k)', 'UCSD OCT (2k)', 'UCSD OCT (2k)'],
    'Model': ['Supervised Pretraining', 'Self-Supervised Pretraining', 'Supervised Pretraining', 'Self-Supervised Pretraining', 'Supervised Pretraining', 'Self-Supervised Pretraining'],
    'mAUC': [0.880, 0.867, 0.968, 0.966, 0.968, 0.966],
    'Accuracy': [0.71, 0.71, 0.86, 0.94, 0.89, 0.92],
    'Precision': [0.71, 0.73, 0.86, 0.95, 0.93, 0.93],
    'Sensitivity': [0.71, 0.71, 0.86, 0.94, 0.89, 0.92],
    'F1-score': [0.71, 0.71, 0.85, 0.94, 0.90, 0.92]
}

# Convert data to DataFrame
df = pd.DataFrame(data)

# Categories for the radar plot
categories = ['mAUC', 'Accuracy', 'Precision', 'Sensitivity', 'F1-score']

# Create a plot
fig, axs = plt.subplots(1, 3, figsize=(18, 8), subplot_kw=dict(polar=True))

# Loop through the datasets and plot radar charts
for i, dataset in enumerate(df['Dataset'].unique()):
    # Filter data for the specific dataset
    dataset_data = df[df['Dataset'] == dataset]
    
    # Plot radar chart for the current dataset (comparing the two models)
    create_radar_chart(dataset_data, categories, ['Supervised Pretraining', 'Self-Supervised Pretraining'], axs[i])

# Adjust layout and display
plt.tight_layout()
plt.savefig('./ssp_retinaloct_tbme/vit2spn_tiny/plotting/ssp_sp_comparison.pdf', dpi=300)
plt.show()
