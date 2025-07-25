import io
from contextlib import nullcontext

import numpy as np

# NOTE: The last line in exported csv is empty due to the '\n' at the last iteration.

def read_csv(file_or_path, columns, header, datarow, delimiter, encoding=None):
    if isinstance(file_or_path, io.IOBase):
        file_or_path.seek(0)
        contextmanager = nullcontext(file_or_path)
    else:
        contextmanager = open(file_or_path, "r", encoding=encoding)
    
    data = {}

    icolumns = {}
    for k, v in columns.items():
        for i in v:
            icolumns[i] = k
    
    columnindices = list(icolumns)

    iheader = {}
    for k, v in header.items():
        iheader[v] = k
    
    lastheaderline = max(iheader)

    with contextmanager as fileobj:
        header_data = {}
        for linenumber, line in enumerate(fileobj):
            if linenumber in iheader:
                key = iheader[linenumber]
                cells = [x.strip() for x in line.split(delimiter)]
                header_data[key] = [cells[x] for x in columnindices]
            if linenumber == lastheaderline:
                break

        fileobj.seek(0)

        raw_np_data = np.genfromtxt(
            fileobj,
            delimiter=delimiter,
            skip_header=datarow,
            usecols=columnindices,
            dtype=None,  # This allows different data types for each column
            encoding=encoding
        )

        np_data = {}
        for k, v in columns.items():
            d = []
            for i in v:
                index = columnindices.index(i)
                try:
                    d.append(raw_np_data[f"f{index}"])
                except:
                    print("!!! WARNING !!! the program find some issue while loading some data. \n Using alternative method to overcome") ###
                    d.append(raw_np_data[:, index])
                
                
            np_data[k] = np.array(d)
        del raw_np_data

        for k, v in columns.items():
            d = {}
            for k2, v2 in header_data.items():
                d[k2] = []
                for a in v:
                    d[k2].append(v2[columnindices.index(a)])
            d["data"] = np_data[k]
            data[k] = d

    return data

def export_csv(file_name, data, names = None, units = None, dummy = "", delimiter=","):

    try:
        m,n = np.shape(data)
    except:
        m = np.shape(data)[0]
        n = 1
    
    _names = ''
    for i in range(n):
        if names == None:
            _n = "H" + str(i+1)
        else:
            _n = str(names[i])

        if i == n-1:
            _names = _names + _n
        else:
            _names = _names + _n + delimiter

    
    _units = ''
    for i in range(n):
        if units == None:
            _u = "u" + str(i+1)
        else:
            _u = str(units[i])

        if i == n-1:
            _units = _units + _u
        else:
            _units = _units + _u + delimiter

    with open(file_name+'.csv', 'w') as f:

        f.write(_names + '\n')
        f.write(_units + '\n')

        if n == 1 :
            for j in range(m):
                if np.isnan(data[j]):
                    f.write(str(dummy) + '\n')
                else:
                    f.write(str(data[j][0]) + '\n')
        else:
            for j in range(m):
                line = ''
                for i in range(n):
                    if i == n-1:
                        if np.isnan(data[j,i]):
                            line = line + str(dummy) # I believe that there is a better way to do that, but at the moment lets try it
                        else:
                            line = line + str(data[j,i])
                    elif np.isnan(data[j,i]):
                        line = line + str(dummy) + delimiter
                    else:
                        line = line + str(data[j,i]) + delimiter
                f.write(line + '\n')