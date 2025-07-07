import pandas as pd
import re
from io import StringIO

def parse_las_file(filepath):
    data_sections = {}
    current_key = None
    current_data_lines = []

    with open(filepath, 'r', encoding='utf-8') as file:
        for line in file:
            stripped = line.strip()

            # Skip blank lines and comments
            if not stripped or stripped.startswith('#'):
                continue

            if stripped.startswith('~'):
                # Save previous section
                if current_key and current_data_lines:
                    df = parse_data_block(current_data_lines)
                    data_sections[current_key] = df
                    current_data_lines = []

                current_key = stripped.lstrip('~').strip()
            else:
                current_data_lines.append(stripped)

        # Add the final section
        if current_key and current_data_lines:
            df = parse_data_block(current_data_lines)
            data_sections[current_key] = df

    return data_sections

def parse_data_block(lines):
    structured_data = []

    # First parse with default (tab/space/colon) logic
    for line in lines:
        line = line.rstrip()
        if not line.strip():
            continue

        parts = re.split(r'(?:\t| {4,})', line, maxsplit=1)

        if len(parts) == 2:
            key = parts[0].strip()
            rest = parts[1].strip()
            rest_parts = re.split(r'\s*:\s*', rest, maxsplit=1)

            if len(rest_parts) == 2:
                structured_data.append([key, rest_parts[0].strip(), rest_parts[1].strip()])
            else:
                structured_data.append([key, rest.strip(), ""])
        else:
            structured_data.append([parts[0].strip(), "", ""])

    # Create initial DataFrame and drop empty columns
    df_original = pd.DataFrame(structured_data)
    df_original = df_original.loc[:, ~df_original.apply(lambda col: col.astype(str).str.strip().eq('').all(), axis=0)]
    n_original_cols = df_original.shape[1]

    # Try parsing with comma separator
    comma_data = [line.split(",") for line in lines if line.strip()]
    df_comma = pd.DataFrame(comma_data)
    df_comma = df_comma.loc[:, ~df_comma.apply(lambda col: col.astype(str).str.strip().eq('').all(), axis=0)]

    # Use the comma-parsed version only if it has more columns
    if df_comma.shape[1] > n_original_cols:
        return df_comma
    else:
        return df_original