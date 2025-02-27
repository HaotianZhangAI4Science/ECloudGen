import os

class load_file:
    VALID_MODES = ["rb", "r"]
    """Open a local file for reading."""

    def __init__(self, file_path, mode, verbose=True):
        if mode not in self.VALID_MODES:
            raise ValueError(f'"{mode}" not in {self.VALID_MODES}')
        self.file_path = file_path
        self.mode = mode
        self.file = None
        self.verbose = verbose

    def __enter__(self):
        if not os.path.isfile(self.file_path):
            raise FileNotFoundError(f"No such file: {self.file_path}")
        
        self.file = open(self.file_path, self.mode)
        return self.file

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.file is not None:
            self.file.close()
