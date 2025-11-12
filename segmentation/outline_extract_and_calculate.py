import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt

# ---------------- Configuration ---------------- #
# Parent directory containing subdirectories for each image's masks.
mask_parent_dir = Path(r"D:\fly\SEG_all_new")  # This is the directory where your original code saved each mask subdirectory

# Folder to save the wing outlines ("broader" images)
outline_output_dir = Path(r"D:\fly\Wing_Outlines_new")
outline_output_dir.mkdir(parents=True, exist_ok=True)

# Output results (TXT and Excel)
results_txt_path = Path(r"D:\fly\Wing_Outlines_new\wing_body_parameters.txt")
results_excel_path = Path(r"D:\fly\Wing_Outlines_new\wing_body_parameters.xlsx")


# ---------------- Helper Functions ---------------- #
def get_largest_contour(binary_img):
    """
    Given a binary image, find and return the largest contour.
    """
    contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    return largest


def compute_wing_parameters(contour):
    """
    Compute wing parameters from a given contour.
    Returns:
        b: wing span (maximum distance between convex hull points)
        S: wing area (using contour area)
        c: average chord (S / b)
        AR: aspect ratio (b^2 / S)
    """
    if contour is None or cv2.contourArea(contour) == 0:
        return 0, 0, 0, 0
    # Compute convex hull of contour points
    hull = cv2.convexHull(contour)
    hull_points = hull.reshape(-1, 2)

    # Wing span: maximum Euclidean distance among hull points
    b = 0
    num_points = len(hull_points)
    for i in range(num_points):
        for j in range(i + 1, num_points):
            d = np.linalg.norm(hull_points[i] - hull_points[j])
            if d > b:
                b = d

    # Wing area S from contour area
    S = cv2.contourArea(contour)

    # Average chord length
    c = S / b if b != 0 else 0
    # Aspect ratio = b / c = b^2 / S
    AR = (b ** 2) / S if S != 0 else 0
    return b, S, c, AR


def compute_body_parameters(wing_span):
    """
    Given wing span (b) as wing length, compute body parameters.
    Returns a dictionary with all parameters.
    """
    # Head dimensions
    head_length = 0.2 * wing_span
    head_width = 0.18 * wing_span
    head_volume = np.pi * (head_width / 2) ** 2 * head_length

    # Thorax dimensions
    thorax_length = 0.2 * wing_span
    thorax_width = 0.2 * wing_span
    thorax_volume = np.pi * (thorax_width / 2) ** 2 * thorax_length

    # Abdomen dimensions
    abdomen_length = 0.5 * wing_span
    abdomen_width = 0.2 * wing_span
    abdomen_volume = np.pi * (abdomen_width / 2) ** 2 * abdomen_length

    total_body_volume = head_volume + thorax_volume + abdomen_volume

    # Muscle mass estimated as water density (assumed 1) * thorax volume
    muscle_mass = thorax_volume

    # Flight energy efficiency: thorax mass / total mass
    flight_efficiency = thorax_volume / total_body_volume if total_body_volume != 0 else 0

    return {
        "Head_Length": head_length,
        "Head_Width": head_width,
        "Head_Volume": head_volume,
        "Thorax_Length": thorax_length,
        "Thorax_Width": thorax_width,
        "Thorax_Volume": thorax_volume,
        "Abdomen_Length": abdomen_length,
        "Abdomen_Width": abdomen_width,
        "Abdomen_Volume": abdomen_volume,
        "Total_Body_Volume": total_body_volume,
        "Muscle_Mass": muscle_mass,
        "Flight_Efficiency": flight_efficiency
    }


def save_outline_image(contour, shape, save_path):
    """
    Draws the contour (outline) on a white background image and saves it.
    The outline is drawn in black.
    """
    outline_img = np.ones(shape, dtype=np.uint8) * 255  # white background
    cv2.drawContours(outline_img, [contour], -1, (0), thickness=2)  # black outline
    cv2.imwrite(str(save_path), outline_img)


def combine_masks(mask_list):
    """
    Given a list of binary mask arrays (all assumed same shape),
    return the union (bitwise OR) of these masks.
    """
    combined = np.zeros_like(mask_list[0], dtype=np.uint8)
    for mask in mask_list:
        combined = cv2.bitwise_or(combined, mask)
    return combined


# ---------------- Main Processing ---------------- #
def main():
    # Find all subdirectories in mask_parent_dir that match the pattern "*_masks"
    mask_subdirs = [d for d in mask_parent_dir.iterdir() if d.is_dir() and d.name.endswith("_masks")]
    print(f"Found {len(mask_subdirs)} mask subdirectories.")

    results = []

    for subdir in tqdm(mask_subdirs, desc="Processing each image directory"):
        # The original image name is derived by removing the "_masks" suffix
        base_name = subdir.name.replace("_masks", "")
        mask_files = list(subdir.glob("*.png"))

        if len(mask_files) == 0:
            continue

        individual_params = []
        mask_list = []
        outline_paths = []

        for f in mask_files:
            # Read the mask in grayscale
            mask_img = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
            if mask_img is None:
                continue
            # Binarize the mask image (in case of any interpolation artifacts)
            _, binary = cv2.threshold(mask_img, 127, 255, cv2.THRESH_BINARY)

            mask_list.append(binary)

            # Get largest contour from the binary mask
            contour = get_largest_contour(binary)
            if contour is None:
                continue
            # Compute wing parameters from this contour
            b, S, c, AR = compute_wing_parameters(contour)

            # Save outline image for this mask
            outline_file = outline_output_dir / f"{f.stem}_outline.png"
            save_outline_image(contour, binary.shape, outline_file)
            outline_paths.append(str(outline_file))

            individual_params.append({
                "Mask_File": str(f),
                "Wing_Span": b,
                "Wing_Area": S,
                "Average_Chord": c,
                "Aspect_Ratio": AR
            })

        # For the combined wing (if more than one mask exists or even one)
        if len(mask_list) > 0:
            combined_mask = combine_masks(mask_list)
            # Get largest contour from the combined mask
            combined_contour = get_largest_contour(combined_mask)
            if combined_contour is not None:
                b_comb, S_comb, c_comb, AR_comb = compute_wing_parameters(combined_contour)
                # Save combined outline image
                combined_outline_path = outline_output_dir / f"{base_name}_combined_outline.png"
                save_outline_image(combined_contour, combined_mask.shape, combined_outline_path)
            else:
                b_comb, S_comb, c_comb, AR_comb = 0, 0, 0, 0
                combined_outline_path = ""
        else:
            b_comb, S_comb, c_comb, AR_comb = 0, 0, 0, 0
            combined_outline_path = ""

        # Use the combined wing span (b_comb) as the wing length for body parameters.
        body_params = compute_body_parameters(b_comb)

        # Prepare a record for this original image
        record = {
            "Image_Name": base_name,
            "Num_Masks": len(mask_files),
            "Combined_Wing_Span": b_comb,
            "Combined_Wing_Area": S_comb,
            "Combined_Average_Chord": c_comb,
            "Combined_Aspect_Ratio": AR_comb,
            "Combined_Outline_Path": str(combined_outline_path),
            "Individual_Params": individual_params
        }
        record.update(body_params)
        results.append(record)

    # Save summary results to TXT and Excel.
    summary_list = []
    for rec in results:
        summary = {
            "Image_Name": rec["Image_Name"],
            "Num_Masks": rec["Num_Masks"],
            "Wing_Span": rec["Combined_Wing_Span"],
            "Wing_Area": rec["Combined_Wing_Area"],
            "Average_Chord": rec["Combined_Average_Chord"],
            "Aspect_Ratio": rec["Combined_Aspect_Ratio"],
            "Head_Length": rec["Head_Length"],
            "Head_Width": rec["Head_Width"],
            "Head_Volume": rec["Head_Volume"],
            "Thorax_Length": rec["Thorax_Length"],
            "Thorax_Width": rec["Thorax_Width"],
            "Thorax_Volume": rec["Thorax_Volume"],
            "Abdomen_Length": rec["Abdomen_Length"],
            "Abdomen_Width": rec["Abdomen_Width"],
            "Abdomen_Volume": rec["Abdomen_Volume"],
            "Total_Body_Volume": rec["Total_Body_Volume"],
            "Muscle_Mass": rec["Muscle_Mass"],
            "Flight_Efficiency": rec["Flight_Efficiency"],
            "Combined_Outline_Path": rec["Combined_Outline_Path"]
        }
        summary_list.append(summary)

    df = pd.DataFrame(summary_list)
    df.to_excel(results_excel_path, index=False)

    # Also write a simple TXT report
    with open(results_txt_path, "w") as f:
        for rec in results:
            f.write(f"Image: {rec['Image_Name']}\n")
            f.write(f"  Number of Masks: {rec['Num_Masks']}\n")
            f.write(f"  Combined Wing Span (b): {rec['Combined_Wing_Span']:.2f}\n")
            f.write(f"  Combined Wing Area (S): {rec['Combined_Wing_Area']:.2f}\n")
            f.write(f"  Average Chord (c = S/b): {rec['Combined_Average_Chord']:.2f}\n")
            f.write(f"  Aspect Ratio (b^2/S): {rec['Combined_Aspect_Ratio']:.2f}\n")
            f.write("  Body Parameters:\n")
            f.write(
                f"    Head: length={rec['Head_Length']:.2f}, width={rec['Head_Width']:.2f}, volume={rec['Head_Volume']:.2f}\n")
            f.write(
                f"    Thorax: length={rec['Thorax_Length']:.2f}, width={rec['Thorax_Width']:.2f}, volume={rec['Thorax_Volume']:.2f}\n")
            f.write(
                f"    Abdomen: length={rec['Abdomen_Length']:.2f}, width={rec['Abdomen_Width']:.2f}, volume={rec['Abdomen_Volume']:.2f}\n")
            f.write(f"    Total Body Volume: {rec['Total_Body_Volume']:.2f}\n")
            f.write(f"    Muscle Mass: {rec['Muscle_Mass']:.2f}\n")
            f.write(f"    Flight Efficiency (thorax mass/total mass): {rec['Flight_Efficiency']:.2f}\n")
            f.write(f"  Combined Outline Saved at: {rec['Combined_Outline_Path']}\n")
            f.write("-" * 50 + "\n")

    print("Processing complete. Results saved to TXT and Excel files.")


if __name__ == "__main__":
    main()
