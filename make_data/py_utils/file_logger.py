class FileLogger(object):
    def __init__(self, full_filename, headers):
        self._headers = headers
        self._full_filename = full_filename
        self._out_fp = open(self._full_filename, 'a')    

    def write(self, line):
        assert len(line) == len(self._headers)
        self._write(line)
        

    def close(self):
        self._out_fp.close()

    def _write(self, arr):
        self._read_fp = open(self._full_filename, 'r')
        lines = self._read_fp.readlines()
        if not lines:
            headers = [str(e) for e in self._headers]
            self._out_fp.write(' '.join(headers) + '\n')
            self._out_fp.flush() 
        arr = [str(e) for e in arr]
        self._out_fp.write(' '.join(arr) + '\n')
        self._out_fp.flush()
    
    def get_info(self):
        self._read_fp = open(self._full_filename, 'r')
        lines = self._read_fp.readlines()
        if not lines:
            self._write(self._headers) 
            return -1, "", 99999., 1. 
        else:  
            last_line = lines[-1].strip().split()
        if last_line[0] == 'curr_epoch':
            return -1, "", 99999., 1.
        else:    
            return int(last_line[0]), str(last_line[-1]), float(last_line[3]), float(last_line[4])