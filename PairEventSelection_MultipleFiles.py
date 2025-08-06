import gzip
import os
import argparse

# Opening the input .sim or .sim.gz file
def open_file(filename, mode):
    if filename.endswith('.gz'):
        return gzip.open(filename, mode, encoding='utf-8')
    elif filename.endswith('.sim'):
        return open(filename, mode, encoding='utf-8')
    else:
        raise ValueError('Unsupported file format.')

# Identifying the pair events
def pair_events(event_lines):
    for line_number, line in enumerate(event_lines):
        if line.startswith('IA INIT'):
            return event_lines[line_number + 1].startswith('IA PAIR')

# Getting the file names from the concatenation file
def extract_concatenation_information(concat_filename):
    input_files = []
    header_lines = []
    with open_file(concat_filename, 'rt') as f:
        for line in f:
            stripped = line.strip()

            # Getting the names of the sim files containing MC data
            if stripped.startswith('IN '):
                input_files.append(stripped[3:])
            elif stripped == 'EN':
                break
            else:
                header_lines.append(line)
    return header_lines, input_files

# Get final lines from file
def extract_footer_lines(file_path):
    # Extracting the end lines (EN, TE, TS) from the last input file
    end_lines = []
    with open_file(file_path, 'rt') as f:
        for line in f:
            if line.strip().startswith(('EN', 'TE', 'TS')):
                end_lines.append(line)
    return end_lines

# Extracting the pair events
def extract_pair_events(input_file):
    with open_file(input_file, 'rt') as f:
        current_event = []

        for line in f:
            stripped = line.strip()

            if stripped.startswith(('EN', 'TE', 'TS')):
                continue

            if stripped == 'SE':
                if current_event and pair_events(current_event):
                    yield 'SE\n'
                    yield from current_event
                current_event = []
            else:
                current_event.append(line)

        # Get last event
        if current_event and pair_events(current_event):
            yield 'SE\n'
            yield from current_event

# Processing the concatenation file
def process_concat_file(concat_filename, output_filename=None):
    header_lines, input_files = extract_concatenation_information(concat_filename)

    if output_filename is None:
        base = os.path.splitext(concat_filename)[0]
        output_filename = f"{base}_PairEventsOnly.sim.gz"
        counter = 2
        while os.path.exists(output_filename):
            output_filename = f"{base}_PairEventsOnly{counter}.sim.gz"
            counter += 1

    def resolve_sim_path(file_path):
        if os.path.exists(file_path):
            return file_path
        elif os.path.exists(file_path + '.gz'):
            return file_path + '.gz'
        else:
            raise FileNotFoundError(f"Could not find file: {file_path} or {file_path}.gz")

    resolved_input_files = [resolve_sim_path(f) for f in input_files]
    footer_lines = extract_footer_lines(resolved_input_files[-1])

    with gzip.open(output_filename, 'wt', encoding='utf-8') as output:
        for line in header_lines:
            output.write(line)
        output.write('TB 0\n')

        for sim_file in resolved_input_files:
            for event_line in extract_pair_events(sim_file):
                output.write(event_line)

        for line in footer_lines:
            output.write(line)

    print(f"All pair events written to: {output_filename}")
    return output_filename

def main(concat_file=None):
    if concat_file is None:
        parser = argparse.ArgumentParser(description='Select pair events from concatenation file referencing SIM files.')
        parser.add_argument('concat_file', type=str, help='Path to the concatenation SIM file')
        args = parser.parse_args()
        concat_file = args.concat_file

    return process_concat_file(concat_file)

if __name__ == '__main__':
    main()
