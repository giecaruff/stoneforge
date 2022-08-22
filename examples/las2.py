import re
import numpy as np
import io


class LAS2Error(Exception):
    pass


_default_line_format_format = (
    "{{mnemonic:<{mnemonic}}}."
    "{{unit:<{unit}}} "
    "{{value:<{value}}} : "
    "{{description:<{description}}}"
)

_default_cell_format = "{:<8.4f}"

_line_regex = re.compile(
    r"(?P<mnemonic>[^\.]+)\.(?P<unit>\S*)(?P<value>.*):(?P<description>.*)"
)

print(_line_regex)

_line_elements = ["mnemonic", "unit", "value", "description"]

_sections = {
    "V": "version",
    "W": "well",
    "P": "parameter",
    "C": "curve",
    "O": "other",
    "A": "data",
}

_sections_order = ["version", "well", "parameter", "curve", "other", "data"]


def _get_null_value(sections):
    nullstring = None
    for line in sections["well"]:
        if line["mnemonic"] == "NULL":
            nullstring = line["value"]
            break
    return float(nullstring)


def _parse_line(line):
    match = _line_regex.match(line)

    if match is None:
        raise LAS2Error("'{}' is not a valid LAS 2.0 line.".format(line))

    parsed_lines = {k: v.strip() for k, v in match.groupdict().items()}
    #parsed_lines.replace(/[^a-zA-Z0-9 ]/g, "")
    #print(parsed_lines)

    return parsed_lines


def _parse_section(lines, previous_sections):
    return [_parse_line(line) for line in lines]


def _parse_plain_text_section(lines, previous_sections):
    return lines


def _parse_data_section(lines, previous_sections):
    ncols = len(previous_sections["curve"])
    nullvalue = _get_null_value(previous_sections)

    data = np.array(" ".join(lines).split(), dtype=float)
    data[data == nullvalue] = np.nan
    data = data.reshape((-1, ncols)).transpose()

    return data


_parsers = {
    "version": _parse_section,
    "well": _parse_section,
    "parameter": _parse_section,
    "curve": _parse_section,
    "other": _parse_plain_text_section,
    "data": _parse_data_section,
}


def read(lasfile):
    """Reads the contents of a LAS 2.0 file.

    Parameters
    ----------
    lasfile : string or file-like object
        The path of the file to read or an existing file-like object to read from.

    Returns
    -------
    dict
        A dictionary containing the sections of the LAS file.

    Notes
    -----
    The structure of the returned dictionary is specified below.
    The dictionary keys are the section names: 'version', 'well', 'parameter', 'curve', 'other', 'data'.
    Not all sections must be present on a LAS 2.0 file.
    For more information on the contents of each section, please refer to the LAS 2.0 standard [1]_.

    The value of the 'data' section is a numpy ndarray where each row contains the data for a well log.

    The value of the 'other' section is a list of lines exactly as found on the original file.

    For all other sections, the values are dictionaries containing four keys: 'mnemonic', 'unit', 'value' and
    'description'.
    For information on the structure of a LAS 2.0 line, please also refer to its specification [1]_.

    References
    ----------
    .. [1] LAS 2.0 standard - http://www.cwls.org/wp-content/uploads/2017/02/Las2_Update_Feb2017.pdf -
       Retrieved August 14, 2019

    Examples
    --------
    The examples below contains ficticious data.
    In the first example we see the version information for the file.
    >>> import las2
    >>> lasfile = las2.read('path/to/the/las/file')
    >>> lasfile['version'][0]
    {'mnemonic': 'VERS', 'unit': '', 'value': '2.00', 'description': 'CWLS LOG ASCII STANDARD - VERSION 2.00'}

    Here we print the names and units for each of the well logs (note that 'DEPTH' is read as a well log).
    >>> for curve_info in lasfile['curve']:
    ...     print("{mnemonic} ({unit})".format(**curve_info))
    DEPTH (M)
    GR (API)
    ...

    The data section where each row contains the values for a log (first row is depth, second is GR, etc...)
    >>> lasfile['data']
    array([[1000.0, 1000.2, ..., 1100.0],
           [25.0,     26.0, ...,   75.0],
           ...]])
    """
    sections = {}
    current_section_key = ""
    current_section = []

    if isinstance(lasfile, io.IOBase):
        lasfile.seek(0)
        close_file = False
    else:
        lasfile = open(lasfile, "r")
        close_file = True

    for line in lasfile:
        if line.lstrip().startswith("#"):
            continue
        elif line.lstrip().startswith("~"):
            _, section_title = line.split("~", 1)
            sections[current_section_key] = current_section
            current_section_key = _sections[section_title[0].upper()]
            current_section = []
        else:
            current_section.append(line)
    sections[current_section_key] = current_section

    if close_file:
        lasfile.close()

    del sections[""]

    parsed_sections = {}

    for section_key in sections:
        parser = _parsers[section_key]
        section = sections[section_key]
        parsed_sections[section_key] = parser(section, parsed_sections)

    return parsed_sections


def _compose_line(line, format):
    return format.format(**line)


def _compose_section(lines, format, previous_sections):
    return [_compose_line(line, format).rstrip() for line in lines]


def _compose_plain_text_section(lines, format, previous_sections):
    return lines


def _compose_data_section(data, format, previous_sections):
    nullvalue = _get_null_value(previous_sections)

    data_section = []
    for i in range(data.shape[1]):
        nanfreeline = data[:, i]
        nanfreeline[np.isnan(nanfreeline)] = nullvalue
        data_section.append(format.format(*nanfreeline).rstrip())

    return data_section


_composers = {
    "version": _compose_section,
    "well": _compose_section,
    "parameter": _compose_section,
    "curve": _compose_section,
    "other": _compose_plain_text_section,
    "data": _compose_data_section,
}


def _section_title_getter(key, section):
    return "~" + key.upper()


def _data_title_getter(key, section):
    return "~A"


_default_section_title_getters = {
    "version": _section_title_getter,
    "well": _section_title_getter,
    "parameter": _section_title_getter,
    "curve": _section_title_getter,
    "other": _section_title_getter,
    "data": _data_title_getter,
}


def _section_format_getter(section):
    maxwidths = dict.fromkeys(_line_elements, 0)
    for line in section:
        for key in line:
            if len(line[key]) > maxwidths[key]:
                maxwidths[key] = len(line[key])

    return _default_line_format_format.format(**maxwidths)


def _plain_text_format_getter(section):
    return "{}"


def _data_format_getter(section):
    n = section.shape[0]
    return " ".join([_default_cell_format] * n)


_default_section_format_getters = {
    "version": _section_format_getter,
    "well": _section_format_getter,
    "parameter": _section_format_getter,
    "curve": _section_format_getter,
    "other": _plain_text_format_getter,
    "data": _data_format_getter,
}


def write(lasfile, data, section_titles=None, section_formats=None):
    """Writes well log data to a file using the LAS 2.0 format.

    Parameters
    ----------
    lasfile : string or file-like object
        The path of the file to read or an existing file-like object to read from.
    data : dict
        A dictionary with the same structure as returned by the `read` function.
    section_titles : dict, optional
        A dictionary where the key is the section name and value is the title that will be used at the beggining
        of the LAS 2.0 file section. For further information please refer to the Notes section.
    section_formats : dict, optional
        A dictionary where the key is the section name and value is the format string that will be used to format the
        lines in the respective section. For further information please refer to the Notes section.

    Notes
    -----
    This function does not guarantee that the output file will follow the LAS 2.0 standard. If mandatory sections or
    lines are missing from the inputs, the file will be written nevertheless. The only required field for the function
    work is the 'NULL' value in the 'well' section.
    Also, no checks are made to guarantee the validity of section titles or formats.

    Possible section names are: 'version', 'well', 'parameter', 'curve', 'other', 'data'.

    Default section titles are '~' followed by the section name (for instance '~VERSION', for the 'version' section),
    except for the 'data' section, which defaults to '~A'.
    Each section title can be individually omitted.
    For more information on the rules for section titles, please refer to the LAS 2.0 standard [1]_.

    Format strings for 'version', 'well', 'parameter' and 'curve' must contain the following fields: 'mnemonic', 'unit',
    'value' and 'description'. Here is an example of a valid format: "{mnemonic}.{unit} {value} : {description}". The
    default format left align fields in the same section within columns with the same width.
    For more information on the construction of valid LAS 2.0 lines, please refer to the LAS 2.0 standard [1]_.
    For the 'other' section, the format string is simply the format of each line of this section. For example, "{}"
    will output the lines as is, which is the default value for this section.
    The format string for the 'data' section contains a column for each well log. For example, in the case of 3 well
    logs, "{:>8.4f} {:>8.4f} {:>8.4f}" is the default format.
    Each section format can be individually omitted.

    See Also
    --------
    read : Reads the contents of a LAS 2.0 file.

    References
    ----------
    .. [1] LAS 2.0 standard - http://www.cwls.org/wp-content/uploads/2017/02/Las2_Update_Feb2017.pdf -
       Retrieved August 14, 2019

    Examples
    --------
    Minimal example of usage. This would not produce a valid LAS 2.0 file since it is missing many of the mandatory
    sections and lines.
    >>> import las2
    >>> from io import StringIO
    >>> data = {}
    >>> data['well'] = [{'mnemonic': 'NULL', 'unit': '', 'value': '-999.0', 'description': ''}]
    >>> data['curve'] = [
    ...     {'mnemonic': 'DEPT', 'unit': 'M', 'value': '', 'description': ''},
    ...     {'mnemonic': 'GR', 'unit': 'API', 'value': '', 'description': ''}
    ... ]
    >>> data['data'] = np.array([
    ...     [1000.0, 1000.2, 1000.4, 1100.0],
    ...     [25.0, 26.0, np.nan, 75.0]
    ... ])
    >>> lasfile = StringIO()
    >>> las2.write(lasfile, data)
    >>> print(lasfile.getvalue())
    ~WELL
    NULL. -999.0 :
    ~CURVE
    DEPT.M    :
    GR  .API  :
    ~A
    1000.0000 25.0000
    1000.2000 26.0000
    1000.4000 -999.0000
    1100.0000 75.0000
    """
    if isinstance(lasfile, io.IOBase):
        lasfile.seek(0)
        close_file = False
    else:
        lasfile = open(lasfile, "w")
        close_file = True

    if section_titles is None:
        section_titles = {}
    for key, section in data.items():
        if key not in section_titles:
            section_titles[key] = _default_section_title_getters[key](key, section)

    if section_formats is None:
        section_formats = {}
    for key, section in data.items():
        if key not in section_formats:
            section_formats[key] = _default_section_format_getters[key](section)

    lines = []
    for key in _sections_order:
        if key not in data:
            continue
        composer = _composers[key]
        format = section_formats[key]
        lines.append(section_titles[key])
        lines.extend(composer(data[key], format, data))
    lasfile.write("\n".join(lines))

    if close_file:
        lasfile.close()
