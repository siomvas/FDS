import json
import matplotlib.pyplot as plt
from PIL import Image
import os
from collections import defaultdict
from utils.argument import args
from utils import helper

# Load input and create output folder
#input_path='/path/to/FDS/data_partitioning/legacy_vlm/data/tiny_imagenet/output_size_test/step3_result.json'
input_path = args.step3_result_path
output_dir = f'{args.output_path}/cluster_visualizations'
os.makedirs(output_dir, exist_ok=True)

# Load step3 output file
data = helper.load_jsonl(input_path)

def clean_llama_output(text, valid_categories):
    """Extract valid category from Llama's verbose output"""
    
    # Ensure text is a string
    if not isinstance(text, str):
        return str(text)
    
    text = text.strip()
    
    # Strategy 1: Check if last line is a valid category (most common case)
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    if lines:
        last_line = lines[-1].strip('"\'.,;')
        # Check if last line exactly matches a category
        for cat in valid_categories:
            if last_line.lower() == cat.lower():
                return cat
    
    # Strategy 2: Search for exact category matches (case-insensitive, whole word)
    import re
    text_lower = text.lower()
    for cat in valid_categories:
        # Use word boundary to match whole words only
        pattern = r'\b' + re.escape(cat.lower()) + r'\b'
        if re.search(pattern, text_lower):
            return cat
    
    # Strategy 3: Fuzzy matching (remove spaces, dashes, slashes)
    text_normalized = text_lower.replace(" ", "").replace("/", "").replace("-", "")
    for cat in valid_categories:
        cat_normalized = cat.lower().replace(" ", "").replace("/", "").replace("-", "")
        if cat_normalized in text_normalized:
            return cat
    
    # Strategy 4: If still no match, return the last line as fallback
    if lines:
        return lines[-1].strip('"\'.,;')
    
    return text.strip()

valid_cats = helper.extract_categories(args.step2b_result_path)
# Group images by class
class_groups = defaultdict(list)
for item in data:
    cleaned = clean_llama_output(item['class'], valid_cats)
    #print("Clean: ", cleaned)
    item['class'] = cleaned  
    class_groups[cleaned].append(item)  

# Set number of samples to display per class
samples_per_class = 30

# Create visualization
for class_name, items in class_groups.items():
    # Limit to specified number of samples
    sample_items = items[:samples_per_class]
    
    # Create subplot grid
    n_samples = len(sample_items)
    fig, axes = plt.subplots(1, n_samples, figsize=(4*n_samples, 4))
    
    # Handle single image case
    if n_samples == 1:
        axes = [axes]
    
    fig.suptitle(f'Class: {class_name} (Total: {len(items)} images)', 
                 fontsize=16, fontweight='bold')
    
    # Display each sample
    for idx, (ax, item) in enumerate(zip(axes, sample_items)):
        img_path = item['image']
        
        # Check if image exists
        if os.path.exists(img_path):
            img = Image.open(img_path)
            ax.imshow(img)
            ax.set_title(f"Label: {item['label']}\n{item['description'][:50]}...", 
                        fontsize=10)
            ax.axis('off')
        else:
            ax.text(0.5, 0.5, 'Image not found', 
                   ha='center', va='center', fontsize=12)
            ax.set_title(f"Label: {item['label']}", fontsize=10)
            ax.axis('off')
    
    plt.tight_layout()
    
    # Save the figure
    name = class_name.replace('/', '-')
    output_path = os.path.join(output_dir, f'class_{name}.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {output_path}")
    
    print(f"\nClass '{class_name}': {len(items)} images")
    print(f"Sample labels: {[item['label'] for item in sample_items[:3]]}")
    print("-" * 80)

# Print summary statistics
print("\n" + "="*80)
print("SUMMARY")
print("="*80)
for class_name, items in sorted(class_groups.items()):
    unique_labels = set(item['label'] for item in items)
    print(f"{class_name}: {len(items)} images, {len(unique_labels)} unique labels")
    print(f"  Labels: {', '.join(sorted(unique_labels)[:5])}")
    if len(unique_labels) > 5:
        print(f"  ... and {len(unique_labels) - 5} more")

print(f"\nAll visualizations saved to '{output_dir}/' directory")