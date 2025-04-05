import os
import re
import csv
import sys

# Specify the top-level directory. You can change this as needed.
# For example, if running the script from the directory containing CHILDES:
BASE_DIR = "../data/childs/CHILDES/Eng-NA"

# Regular expression to extract the child_name from the @Participants line.
# We look for a token that begins with "CHI" followed by a name and then "Target_Child".
pattern_child = re.compile(r'@Participants:.*?\bCHI\s+([^ \t]+)\s+Target_Child', re.IGNORECASE)

# Regular expression to capture the @ID line for the child.
# We assume the line has a format like:
#   @ID:    eng|Clark|CHI|2;10.02|male|TD|UC|Target_Child|||
# We want the fourth field, which contains "years;months.days"
# (This regex uses groups on the fourth field.)
pattern_id = re.compile(r'@ID:\s*\S+\|(?:[^|]+\|){1}CHI\|([^|]+)')

# Regular expression for conversation lines. We want those starting with '*'
# followed by one of the allowed speaker codes (CHI, FAT, MOT, INV, KEV)
# followed by a colon then text.
#pattern_conv = re.compile(r'^\*(CHI|FAT|MOT|INV|KEV):\s*(.*)')
pattern_conv = re.compile(r'^\*([A-Z]{3}):\s*(.*)')

def process_file(filepath):
    # Prepare default metadata values (if not found, they will remain empty)
    child_name = ""
    years = ""
    months = ""
    days = ""
    
    # Read the file lines
    try:
        with open(filepath, encoding="utf-8") as f:
            lines = f.readlines()
    except Exception as e:
        sys.stderr.write(f"Error reading file {filepath}: {e}\n")
        return []
    
    # Extract child_name from Participants line (if found)
    for line in lines:
        if line.startswith("@Participants:"):
            m = pattern_child.search(line)
            if m:
                child_name = m.group(1).strip()
                break

    # Extract the @ID line containing CHI info. Use the first matching one.
    for line in lines:
        if line.startswith("@ID:") and "CHI" in line:
            m = pattern_id.search(line)
            if m:
                # The fourth field should be of form "years;months.days"
                id_field = m.group(1).strip()
                # Split by ';'. We expect two parts.
                parts = id_field.split(";")
                if len(parts) == 2:
                    years = parts[0]
                    # Now split the second part by period.
                    subparts = parts[1].split(".")
                    if len(subparts) == 2:
                        months = subparts[0]
                        days = subparts[1]
                break

    # Get file metadata from the path: folder name and root name.
    folder_name = os.path.dirname(filepath)[len(BASE_DIR):].strip("/")
    root_name = os.path.splitext(os.path.basename(filepath))[0]

    # List to hold CSV rows from conversation lines found in this file.
    rows = []

    current_speaker = None
    current_text = ""

    # Open and iterate file lines
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.rstrip("\n")
            m_conv = pattern_conv.match(line)
            if m_conv:
                
                lines = m_conv.group(2).strip().split("&*")
                speaker = m_conv.group(1).strip()
                text = m_conv.group(2).strip()

                for line_index in range(len(lines)):
                    if line_index != 0:
                        speaker = lines[line_index].split(":",1)[0]
                        text = lines[line_index].split(":",1)[1]
                    elif line_index == 0 and len(lines) > 1:
                        speaker = speaker
                        text = lines[line_index]
                    else:
                        speaker = speaker
                        text = text

                    # Clean the text
                    text = clean_text(text)

                    # Skip empty lines after cleaning
                    if not text or text in ["0.", ".", "0"]:
                        continue

                    # Same speaker as previous: combine texts, but check for duplicates
                    if speaker == current_speaker:
                        # Skip if this text is exactly the same as what we already have
                        if text == current_text:
                            continue
                        current_text += " " + text
                    else:
                        # New speaker: save previous (if exists) and start new
                        if current_speaker is not None:
                            rows.append({
                                "folder_name": folder_name,
                                "root_name": root_name,
                                "child_name": child_name,
                                "years": years,
                                "months": months,
                                "days": days,
                                "speaker": current_speaker,
                                "text": current_text
                            })
                        # Start tracking new speaker & text
                        current_speaker = speaker
                        current_text = text

    # Finally, don't forget to add the last accumulated line
    if current_speaker is not None and current_text:
        rows.append({
            "folder_name": folder_name,
            "root_name": root_name,
            "child_name": child_name,
            "years": years,
            "months": months,
            "days": days,
            "speaker": current_speaker,
            "text": current_text
        })

    return rows
def clean_text(text):
    # Step 1: remove [=! text] or [=? text] or [+ text] or [% text]
    #text = re.sub(r'\[=!.*?\]', '', text)
    #remove all text after 
    text = text.split("", 1)[0]

    #remove ⌉ and ⌊ ⌈ ⌋
    text = text.replace("⌉", "").replace("⌊", "").replace("⌈", "").replace("⌋", "")
    
    
    text = re.sub(r'\[(\?|!|\*|>|<){1,3}\]', '', text)

    text = re.sub(r'&[=~\-\+][a-z:_]*', '', text)
    text = re.sub(r'\[[=%+\-\*][^\]]*\]', '', text)

    # Step 2: replace [: word] corrections
    text = re.sub(r'(\S+)\s*\[:\s*(.+?)\]', r'\2', text)
    


    # Step 3: recursive deletion with [/] [//] [///]
    # Handle phrases enclosed in < > and individual words before slashes
    while True:
        prev_text = text
        text = re.sub(r'<[^>]+>\s*\[/{1,3}\]', '', text)
        text = re.sub(r'\S+\s*\[/{1,3}\]', '', text)
        if prev_text == text:
            break

    text = re.sub(r'\[(\?|/|!|\*|>|<){1,3}\]', '', text)

    text = text.replace("<", "").replace(">", "")

    #+// and +"/ and /+" and +" and +... and +.. and +/ and ++ and +, and +. and  + 
    text = text.replace("+//", "").replace("+\"/", "").replace("/+", "").replace("+\"", "").replace("+..", "").replace("+/", "").replace("++", "").replace("+,", "").replace("+.", ".").replace("+^", "")
    
    
    #replace " + " or "+ " beggning of sentance or " +" at end fo sentance
    text = re.sub(r'( |^)\+( |$)', ' ', text)
    text = text.replace("+", " ")# last one


    text = re.sub(r'\((.*?)\)', r'\1', text)

    # New step: remove any remaining content in square brackets
    text = re.sub(r'\[.*?\]', '', text)

    text = re.sub(r'@\S+', '', text)

    text = text.replace(":", "")

    text = text.replace("xxx", " ")

    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'\s+([.,?!])', r'\1', text)


    #remove , at begenning of sentances and any punction series to the last punctiation
    text = re.sub(r'[\.,/!/?]+\.', '.', text)
    text = re.sub(r'[\.,/!/?]+\?', '?', text)
    text = re.sub(r'[\.,/!/?]+!', '!', text)
    text = re.sub(r'[\.,/!/?]+,', ',', text)

    #remove punctioantion followed by whyte space at beggining of sentance
    text = re.sub(r'^[\.,/!/?]+\s*', '', text)



    return text.strip()
    


import csv
from collections import defaultdict

INPUT_CSV = "child_full.csv"
OUTPUT_CSV = "child_response_pairs.csv"

def read_csv(input_csv):
    conversations = defaultdict(list)
    with open(input_csv, mode="r", encoding="utf-8") as f:
        csv_reader = csv.DictReader(f)
        for row in csv_reader:
            conversations[row["root_name"]].append(row)
    return conversations

def pair_speaker_with_chi_response(conversation_rows):
    pairs = []
    for i in range(len(conversation_rows) - 1):
        current_row = conversation_rows[i]
        next_row = conversation_rows[i + 1]
        # we skip CHI lines, looking only for non-CHI speaker followed immediately by CHI
        if current_row["speaker"] != "CHI" and next_row["speaker"] == "CHI":
            pairs.append({
                "folder_name": current_row["folder_name"],
                "root_name": current_row["root_name"],
                "child_name": current_row["child_name"],
                "years": current_row["years"],
                "months": current_row["months"],
                "days": current_row["days"],
                "speaker": current_row["speaker"],
                "text": current_row["text"],
                "CHI_response": next_row["text"]
            })
    return pairs
def futher_csv_processing():
    conversations = read_csv(INPUT_CSV)
    fieldnames = ["folder_name", "root_name", "child_name", 
                  "years", "months", "days", 
                  "speaker", "text", "CHI_response"]

    with open(OUTPUT_CSV, mode="w", newline="", encoding="utf-8") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()

        total_pairs = 0
        for conv_rows in conversations.values():
            pairs = pair_speaker_with_chi_response(conv_rows)
            #if second pair text is lenght less then 3 and start with 0, then skip
            if len(pairs) > 0 and len(pairs[0]['CHI_response']) < 3 and pairs[0]['CHI_response'].startswith('0'):
                continue
            for pair in pairs:
                writer.writerow(pair)
                total_pairs += 1

    print(f"Finished writing to {OUTPUT_CSV}; total pairs found: {total_pairs}")




def main():
    # We'll write the data as CSV to a file called child_full.csv.
    fieldnames = ["folder_name", "root_name", "child_name", "years", "months", "days", "speaker", "text"]
    with open("child_full.csv", mode="w", newline="", encoding="utf-8") as outfile:
        csv_writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        csv_writer.writeheader()
        
        # Recursively search through BASE_DIR for all .cha files.
        #for root, dirs, files in os.walk(BASE_DIR):
        from tqdm import tqdm
        for root, dirs, files in tqdm(os.walk(BASE_DIR)):
            for file in files:
                if file.endswith(".cha"):
                    #if final file start with . then skip
                    if file.startswith('.'):
                        continue
                    filepath = os.path.join(root, file)
                    rows = process_file(filepath)
                    for row in rows:
                        csv_writer.writerow(row)


    


if __name__ == "__main__":
    main()
    futher_csv_processing()