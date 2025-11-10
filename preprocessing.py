import os
import logging
from pathlib import Path
from PIL import Image
import pandas as pd
from tqdm import tqdm

# ---------------------------- Configuration ---------------------------- #

# Define input and output directories
input_dir  = Path(r"D:\fly2\only\00")
output_dir = Path(r"D:\fly2\only\11")

# Create output directory if it doesn't exist
output_dir.mkdir(parents=True, exist_ok=True)

# Desired image size (width, height)
IMAGE_SIZE = (2480, 1748)

# Supported image formats
SUPPORTED_FORMATS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'} # Added .tiff just in case

# Setup logging
logging.basicConfig(
    filename=output_dir / "errors.log",
    filemode='w',
    format='%(asctime)s %(levelname)s:%(message)s',
    level=logging.ERROR
)

# ---------------------------- Processing ---------------------------- #

families, species, names = [], [], []

# --------------------------------------------------------------------- #
# 1) Collect every image in every sub‑folder (1 or more levels deep)    #
# 2) Prefix the original file name with its immediate sub‑folder name   #
# 3) Convert non-grayscale images to grayscale before resizing        # <--- NEW LOGIC
# --------------------------------------------------------------------- #

# Use rglob to walk all sub‑folders
image_files = [
    f for f in input_dir.rglob('*')
    if f.is_file() and f.suffix.lower() in SUPPORTED_FORMATS
]

for file_path in tqdm(image_files, desc="Processing Images"):
    subfolder = file_path.parent.name
    original_stem = file_path.stem

    # Prepend "<subfolder>-" to the logical filename we parse
    filename = f"{subfolder}-{original_stem}"

    # Initialise default values
    family = filename
    species_name = ""
    insect_name = ""

    try:
        # --- family / species / name parsing (unchanged logic) ---
        if '-' in filename:
            parts     = filename.split('-', 1)
            family    = parts[0].strip()
            remainder = parts[1].strip()

            if ' ' in remainder:
                species_name, insect_name = remainder.split(' ', 1)
                species_name = species_name.strip()
                insect_name  = insect_name.strip()
            else:
                species_name = remainder
        else:
            family = filename

        families.append(family)
        species.append(species_name)
        names.append(insect_name)

        # --- resize & save with grayscale conversion ---
        with Image.open(file_path) as img:
            # *** MODIFIED SECTION START ***
            # Check if the image is not already grayscale ('L' mode)
            if img.mode != 'L':
                # Convert to grayscale if it's not
                img = img.convert('L')
            # If it was already 'L', we do nothing and keep it as is.
            # *** MODIFIED SECTION END ***

            # Now resize the image (which is now guaranteed to be 'L' mode)
            img_resized = img.resize(IMAGE_SIZE, Image.Resampling.LANCZOS)

            # Ensure output file name also carries "<subfolder>-"
            output_file_path = output_dir / f"{subfolder}-{original_stem}.png"

            # Save the potentially converted and resized image as PNG
            # PNG supports grayscale ('L') mode perfectly.
            img_resized.save(output_file_path, format='PNG')

    except Exception as e:
        # Log the error with file path and exception details
        logging.error(f"Error processing {file_path}: {str(e)}") # Changed file_path.name to file_path for more context


# ---------------------------- Statistics ---------------------------- #

# Check if any images were processed before creating DataFrame
if families:
    data = pd.DataFrame({
        'Family' : families,
        'Species': species,
        'Name'   : names
    })

    unique_families = data['Family'].nunique()
    unique_species  = data['Species'].nunique()
    unique_names    = data['Name'].nunique() # Note: Names might be empty strings

    # Filter out empty strings before getting value counts and unique lists if desired
    # Example for species:
    # valid_species = data['Species'][data['Species'] != '']
    # species_counts = valid_species.value_counts()
    # unique_species_list = valid_species.unique()

    # Using original logic for now:
    family_counts  = data['Family'].value_counts()
    species_counts = data['Species'][data['Species'] != ''].value_counts() # Count non-empty species
    name_counts = data['Name'][data['Name'] != ''].value_counts() # Count non-empty names

    # ---------------------------- Output Statistics ---------------------------- #

    print("----- Statistics -----")
    print(f"Total Images Processed Successfully: {len(data)}")
    print(f"Number of Unique Families : {unique_families}")
    # Adjust count for potentially empty strings if they shouldn't be counted as unique
    print(f"Number of Unique Species  : {data['Species'][data['Species'] != ''].nunique()}")
    print(f"Number of Unique Names    : {data['Name'][data['Name'] != ''].nunique()}\n")

    print("List of Families:")
    for fam in sorted(data['Family'].unique()): # Sort for better readability
        print(f"- {fam}")

    print("\nList of Species (non-empty):")
    for sp in sorted(data['Species'][data['Species'] != ''].unique()): # Sort and filter empty
        print(f"- {sp}")

    print("\nList of Names (non-empty):")
    for nm in sorted(data['Name'][data['Name'] != ''].unique()): # Sort and filter empty
        print(f"- {nm}")

    # Save statistics to CSV files
    statistics_output = output_dir / "statistics.csv"
    data.to_csv(statistics_output, index=False)
    print(f"\nDetailed data saved to {statistics_output}")

    family_counts_output = output_dir / "family_counts.csv"
    family_counts.to_csv(family_counts_output, header=['Count'], index_label='Family') # Add index label
    print(f"Family counts saved to {family_counts_output}")

    if not species_counts.empty:
        species_counts_output = output_dir / "species_counts.csv"
        species_counts.to_csv(species_counts_output, header=['Count'], index_label='Species') # Add index label
        print(f"Species counts saved to {species_counts_output}")
    else:
        print("No non-empty species found to save counts.")

    # Optionally, save unique lists to separate text files
    with open(output_dir / "unique_families.txt", 'w', encoding='utf-8') as f:
        f.write('\n'.join(sorted(data['Family'].unique())))

    # Save only non-empty unique species and names
    unique_species_list = sorted(data['Species'][data['Species'] != ''].unique())
    if unique_species_list:
        with open(output_dir / "unique_species.txt", 'w', encoding='utf-8') as f:
            f.write('\n'.join(unique_species_list))

    unique_names_list = sorted(data['Name'][data['Name'] != ''].unique())
    if unique_names_list:
         with open(output_dir / "unique_names.txt", 'w', encoding='utf-8') as f:
            f.write('\n'.join(unique_names_list))

    print("Unique lists saved to respective text files.")

else:
    print("No images processed. Check input directory and supported formats.")

print("Processing complete.")