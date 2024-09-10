"""
This script processes captions using the Meta-Llama-3.1-70B-Instruct model to extract relevant objects mentioned in the captions.
Alternatively, if --task_type negative, it generates related negative objects that are not explicitly mentioned in the captions. 
The results are saved to a new CSV file.

source /mnt/nfs_asia/miniconda3/etc/profile.d/conda.sh
conda create -n vLLM python=3.10 -y
export TMPDIR=/mnt/nfs_asia/tmp
export PIP_CACHE_DIR=/mnt/nfs_asia/pip_cache
export HF_HOME="/mnt/nfs_asia/tmp/huggingface"
export HF_TOKEN="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"
conda activate vLLM

Positive Object Extraction:
input_file="/mnt/nfs_asia/csvs/cc12m_images_captions.csv"
output_base="/mnt/nfs_asia/csvs/cc12m_images_extracted_pos"
CUDA_VISIBLE_DEVICES=0,1,2,3 python process_caption_objects.py --input_file $input_file --output_base $output_base --task_type extraction --index_start 0 --index_end 10 > process_output.log 2>&1 &
Negative Objects:
python process_caption_objects.py --input_file /data/healthy-ml/gobi1/data/cc3m/negation_dataset/train_images_extracted_pos.csv --output_base /data/healthy-ml/gobi1/data/cc3m/negation_dataset/train_images_extracted_pos_neg --task_type negative
"""

import argparse
import pandas as pd
from vllm import LLM, SamplingParams
import time

def parse_args():
    parser = argparse.ArgumentParser(description='Extract relevant objects or generate negative objects from captions and save to a CSV file.')
    parser.add_argument('--input_file', type=str, default='captions.csv', help='Path to the input CSV file with captions.')
    parser.add_argument('--index_start', type=int, default=0, help='Start index for caption processing.')
    parser.add_argument('--index_end', type=int, default=-1, help='End index for caption processing.')
    parser.add_argument('--output_base', type=str, required=True, help='Base name for the output CSV file.')
    parser.add_argument('--task_type', type=str, choices=['extraction', 'negative'], required=True, help='Specify the task type: "extraction" or "negative".')

    args = parser.parse_args()
    return args

def process_llm_output(output_text):
    """
    Processes the LLM output to extract the list of relevant objects.

    Handles various edge cases, including:
    - Empty output
    - Unwanted text
    - Incorrect delimiters
    - Repeated objects
    - Malformed responses

    Args:
        output_text (str): The text output by the LLM.

    Returns:
        list: A list of extracted objects. Returns an empty list if the LLM fails to output any objects.
    """
    # Trim any leading or trailing whitespace
    output_text = output_text.strip()

    # Handle empty output
    if not output_text:
        print("Error: LLM did not output any objects.")
        return []

    # Normalize quotes (remove any extra single or double quotes around items)
    output_text = output_text.replace("'", "").replace('"', '')

    # Split the output text into a list of objects by commas
    objects = [obj.strip() for obj in output_text.split(',') if obj.strip()]

    # Handle repeated objects (remove duplicates)
    objects = list(dict.fromkeys(objects))

    # Handle the case where no valid objects were identified
    if not objects:
        print("Error: LLM output is invalid or empty after processing.")
        return []

    return objects

def generate_prompt(caption, relevant_objects=None, prompt_type="extraction"):
    """
    Generate a prompt for either extracting objects or generating negative objects based on the specified prompt type.

    Parameters:
    caption (str): The caption describing the image.
    relevant_objects (list): A list of relevant objects (required if prompt_type is "negative").
    prompt_type (str): The type of prompt to generate. Can be "extraction" or "negative".
    
    Returns:
    str: The generated prompt.
    """
    if prompt_type == "extraction":
        return generate_extraction_prompt(caption)
    elif prompt_type == "negative":
        if relevant_objects is None:
            raise ValueError("relevant_objects must be provided for generating negative objects prompt.")
        return generate_negative_objects_prompt(caption, relevant_objects)
    else:
        raise ValueError("Invalid prompt_type. Must be either 'extraction' or 'negative'.")

def generate_extraction_prompt(caption):
    """
    Generate a prompt for extracting objects based on a caption.
    """
    prompt_template = f"""You are given a caption that describes an image. Your goal is to identify and list the relevant objects or concepts mentioned in the caption. These will be items that could be visually represented in an image.

Instructions:

1. Identify and list each relevant object mentioned in the caption.
2. Always extract at least one noun, even if the caption seems abstract. This could include general concepts, people, animals, places, or objects.
3. Avoid including any descriptive text, quantities, opinions, or additional comments.
4. Provide the output in a clear, comma-separated list of objects.

Here are some examples:

Caption: "a dog playing with a ball in the park"
Output: dog, ball, park

Caption: "a man riding a bicycle near a lake"
Output: man, bicycle, lake

Caption: "a red rose on a white background"
Output: red rose

Caption: "a cat sitting on a chair next to a window"
Output: cat, chair, window

Caption: "a car parked in front of a house with a garden"
Output: car, house, garden

Caption: "{caption}"
Output: """
    return prompt_template

def generate_negative_objects_prompt(caption, relevant_objects):
    """
    Generate a prompt for extracting negative objects based on a caption and a list of relevant objects.
    """
    prompt_template = f"""Task: You are given a caption and a list of relevant objects mentioned in the caption. These objects are items that could be visually represented in an image. Your goal is to generate a new set of negative objects. Negative objects are items that could be related to the relevant objects (or any objects in the caption) but are absent from both the caption and the provided list of relevant objects.

Instructions:

1. Consider the caption and the list of relevant objects.
2. Think of other objects that are related but are not explicitly mentioned in the caption or the provided list.
3. Provide five negative items as a comma-separated list.
4. Do not include any additional text.

Here are some examples:

Caption: "red apple on a wooden table"
Relevant Objects: [red apple, wooden table]
Output: pineapple, knife, cookies, glass, cutting board

Caption: "blue car parked on the street"
Relevant Objects: [blue car, street]
Output: pedestrian, traffic light, bicycle, mailbox, fire truck

Caption: "a brown dog playing with a ball in the park"
Relevant Objects: [brown dog, ball, park]
Output: cat, bench, fountain, leash, swing

Caption: "Snow-covered mountains rise majestically against a clear blue sky"
Relevant Objects: [mountains, snow, sky]
Output: snowboard, cabin, ski poles, pine tree, airplane

Caption: "Freshly baked bread cooling on the kitchen counter"
Relevant Objects: [bread, kitchen counter]
Output: butter, coffee cup, toaster, plate, knife

Caption: "{caption}"
Relevant Objects: {relevant_objects}
Output: """
    return prompt_template

def main(args):
    # Read and process only the specified range of captions
    df = pd.read_csv(args.input_file)
    original_len = len(df)

    # Check if index_end is larger than the number of captions
    if args.index_end == -1 or args.index_end > len(df):
        args.index_end = len(df)  # Set index_end to the number of captions

    # Select the specified range of captions
    df = df.iloc[args.index_start:args.index_end]

    # Filter out rows with empty extracted_objects if task_type is "negative"
    captions = df["caption"].tolist()
    
    # Initialize the LLM detector
    print("Initializing the LLM model...")
    llm = LLM(model="meta-llama/Meta-Llama-3.1-70B-Instruct", tensor_parallel_size=4)
    sampling_params = SamplingParams(temperature=0.8, max_tokens=900, stop=["\n\n"])
    
    # Generate prompts based on the selected task
    if args.task_type == "extraction":
        prompts = [generate_prompt(caption, prompt_type="extraction") for caption in captions]
    elif args.task_type == "negative":
        # relevant_objects_list = df.iloc[args.index_start:args.index_end]["extracted_objects"].tolist()
        relevant_objects_list = df["extracted_objects"].tolist()
        prompts = [generate_prompt(caption, relevant_objects, prompt_type="negative")
                   for caption, relevant_objects in zip(captions, relevant_objects_list)]
    
    start_time = time.time()
    outputs = llm.generate(prompts, sampling_params)
    end_time = time.time()

    extracted_items = []
    failed_count = 0  # Counter for failed outputs

    for output in outputs:
        try:
            generated_text = output.outputs[0].text
            items_part = process_llm_output(generated_text)
            extracted_items.append(items_part)

            if not items_part: # Check if the extraction failed
                failed_count += 1

        except:
            print(f"Error processing output with generated text:\n {generated_text}")
            extracted_items.append([])
            failed_count += 1 # Increment the failed count

    print(f"Generation time (for {len(captions)} prompts): {end_time - start_time} seconds")

    total_prompts = len(captions)
    failure_rate = (failed_count / total_prompts) * 100
    print(f"Failed to extract objects from {failed_count} out of {total_prompts} prompts ({failure_rate:.2f}%).")

    # Add the extracted items to the DataFrame
    output_column = "extracted_objects" if args.task_type == "extraction" else "negative_objects"

    # iloc is used above for extracting captions, and it is exclusive of args.index_end.
    # However, loc is inclusive of args.index_end when selecting rows.
    # To ensure the length of the slice matches the length of extracted_items, we subtract 1 from args.index_end.
    df.loc[args.index_start:args.index_end - 1, output_column] = pd.Series(extracted_items).values

    # Save the updated DataFrame to a new CSV file
    if args.index_start == 0 and args.index_end == original_len:
        output_file_name = f"{args.output_base}.csv"
    else:
        output_file_name = f"{args.output_base}_{args.index_start}_{args.index_end}.csv"
    df.to_csv(output_file_name, index=False)

if __name__ == "__main__":
    args = parse_args()
    main(args)