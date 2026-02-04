'''
This script selects only the pair events converting in dead material from a given .sim or .sim.gz file.
It creates a new output file with the same name as the input file, but with '_DeadMaterialPairEventsOnly' appended before the file extension.
If a file with that name already exists, it appends a counter to the filename to avoid overwriting.
'''
import gzip
import os
import argparse

# Opening the input .sim or .sim.gz file
def open_file (filename, mode):
    if filename.endswith('.gz'):
        return gzip.open(filename, mode, encoding = 'utf-8')
    if filename.endswith('.sim'):
        return open(filename, mode, encoding = 'utf-8')

# Identifying the pair events
def pair_events(event_lines):
    for line_number, line in enumerate(event_lines):
        if line.startswith('IA INIT'):
            return event_lines[line_number + 1].startswith('IA PAIR')

# Selecting only pair events
def select_pair_events(input_filename, output_filename):
    with open_file(input_filename, 'rt') as inputfile, open_file(output_filename, 'wt') as outputfile: 
        header_end = False
        current_event  = []
        end_lines  = []

        for line in inputfile:
            stripped_line = line.strip()
            
            if not header_end:
                outputfile.write(line)
                if stripped_line == 'TB 0':
                    header_end = True
                continue
            
            # For lines at the end of the file:
            if stripped_line.startswith(('EN', 'TE', 'TS')):
                end_lines.append(line)
                continue

            if stripped_line == 'SE':
                if current_event and pair_event_in_detector0(current_event):
                    for event in current_event:
                        outputfile.write(event)
                current_event = [line]
            else:
                current_event.append(line) 

        # This ends by reading 'SE' and processing the event above it -> add one last statement for the last event:
        if current_event and pair_event_in_detector0(current_event):
            for event in current_event:
                outputfile.write(event)
        
        # Putting the last lines back in:
        for end_line in end_lines:
            outputfile.write(end_line)

def pair_event_in_detector0(event_lines):
    for i, line in enumerate(event_lines):
        if line.startswith('IA INIT'):
            if i + 1 < len(event_lines):
                next_line = event_lines[i + 1]
                if next_line.startswith('IA PAIR'):
                    detectorID = next_line.strip().split(';')[2]
                    return detectorID == "0"
    return False

# Main function for execution in terminal
def main():
    parser = argparse.ArgumentParser(description = 'Filtering out the pair events with conversion in dead material from given .sim or .sim.gz file.')
    parser.add_argument('inputfile', type = str, help = 'Attach the file path for the desired input file.')
    args = parser.parse_args()

    inputfile = args.inputfile

    # Declaring the inital file name
    if inputfile.endswith('.sim.gz'):
        base_filename = inputfile[:-7]
        filetype  = '.sim.gz'
    elif inputfile.endswith('.sim'):
        base_filename = inputfile[:-4]
        filetype = '.sim'
    else:
        raise ValueError('Input file must be of type .sim or .sim.gz')

    # Creating an output file with same name as input file except with 'DeadMaterialPairEventsOnly' added
    outputfile = f"{base_filename}_DeadMaterialPairEventsOnly{filetype}"
    counter = 2 
    while os.path.exists(outputfile):
        outputfile = f"{base_filename}_DeadMaterialPairEventsOnly{counter}{filetype}"
        counter += 1

    select_pair_events(inputfile, outputfile)
    print(f"File containing only pair events converting in dead material written to: {outputfile}")

if __name__ == "__main__":
    main()

'''
Proper use in command line: $ python3 PairEventSelection.py <INPUT FILE PATH>
'''

