import os
import pandas as pd
import argparse

def combine_csv_files(base_dir, csv_type):
    """
    Combines multiple CSV files into a single CSV file.

    Args:
        base_dir (str): The base directory where the CSV files are located.
        csv_type (str): The prefix of the CSV files to combine.

    The function reads all CSV files with the specified prefix from the directory,
    combines them, and saves the result as a new CSV file.
    """
    # Define the CSV directory and construct the output file path
    csv_dir = os.path.join(base_dir, "csvs/negation_dataset")
    output_file = os.path.join(csv_dir, f"combined/{csv_type}.csv")

    # Define the indices based on csv_type
    if csv_type == "cc12m_images_extracted_pos":
        indices = [
            (0, 666925),
            (666925, 1333850),
            (1333850, 2000775),
            (2000775, 2667700),
            (2667700, 3334625),
            (3334625, 4001550),
            (4001550, 4668475),
            (4668475, 5335400),
            (5335400, 6002325),
            (6002325, 6669250),
            (6669250, 7336175),
            (7336175, 8003100),
            (8003100, 8670025),
            (8670025, 9336950),
            (9336950, 10003875),
            (10003875, 10003876)  # Last file handles remaining rows
        ]
    elif csv_type == "cc12m_images_extracted_pos_neg":
        indices = [
            (0, 625242),
            (625242, 1250484),
            (1250484, 1875726),
            (1875726, 2500968),
            (2500968, 3126210),
            (3126210, 3751452),
            (3751452, 4376694),
            (4376694, 5001936),
            (5001936, 5627178),
            (5627178, 6252420),
            (6252420, 6877662),
            (6877662, 7502904),
            (7502904, 8128146),
            (8128146, 8753388),
            (8753388, 9378630),
            (9378630, 10003876)  # Last file handles remaining rows
        ]
    elif csv_type == "cc12m_images_pos_neg_filtered":
        indices = [
            (0, 416828),
            (1250484, 1667312),
            (1667312, 2084140),
            (2084140, 2500968),
            (2500968, 2917796),
            (2917796, 3334624),
            (3334624, 3751452),
            (3751452, 4168280),
            (4168280, 4585108),
            (416828, 833656),
            (4585108, 5001936),
            (5001936, 5418764),
            (5418764, 5835592),
            (5835592, 6252420),
            (6252420, 6669248),
            (6669248, 7086076),
            (7086076, 7502904),
            (7502904, 7919732),
            (7919732, 8336560),
            (833656, 1250484),
            (8753388, 9170216),
            (9170216, 9587044),
            (9587044, -1)  # Last file handles remaining rows
        ]
    else:
        print(f"Error: Unsupported csv_type '{csv_type}'. Please check your input.")
        return

    # List of CSV files to combine based on the provided type
    csv_files = [
        os.path.join(csv_dir, f"{csv_type}_{start}_{end}.csv")
        for start, end in indices
    ]

    # Initialize an empty list to hold dataframes
    df_list = []

    # Read each CSV file and append to the list
    for csv_file in csv_files:
        print(f"Reading {csv_file}...")
        try:
            df = pd.read_csv(csv_file)
            df_list.append(df)
        except FileNotFoundError:
            print(f"Warning: {csv_file} not found. Skipping.")

    # Concatenate all dataframes
    if df_list:
        combined_df = pd.concat(df_list, ignore_index=True)

        # Save the combined dataframe to a new CSV file
        print(f"Saving combined CSV to {output_file}...")
        combined_df.to_csv(output_file, index=False)
        print("All files combined successfully!")
    else:
        print("No files were found to combine.")

def parse_args():
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Combine multiple CSV files into a single CSV file.")
    parser.add_argument(
        '--base_dir', 
        type=str, 
        default="/mnt/nfs_asia", 
        help="Base directory where the CSV files are located (default: /mnt/nfs_asia)"
    )
    parser.add_argument(
        '--csv_type', 
        type=str, 
        default="cc12m_images_extracted_pos", 
        help="Prefix of the CSV files to combine (default: cc12m_images_extracted_pos)"
    )
    return parser.parse_args()

if __name__ == "__main__":
    # Parse command-line arguments
    args = parse_args()

    # Call the function to combine CSV files
    combine_csv_files(args.base_dir, args.csv_type)

    # Example usage:
    # python combine_csv_files.py --csv_type cc12m_images_extracted_pos
    # python combine_csv_files.py --csv_type cc12m_images_extracted_pos_neg
    # python combine_csv_files.py --csv_type cc12m_images_pos_neg_filtered