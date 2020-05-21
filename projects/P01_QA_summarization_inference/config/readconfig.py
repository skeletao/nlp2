import configparser
from pathlib import Path


class ReadConfig:
    """Read and parse config.ini"""

    def __init__(self, file_path=None):
        if file_path:
            conf_path = file_path
        else:
            conf_path = Path(__file__).resolve().parent.as_posix()+ '/config.ini'

        self.cf = configparser.ConfigParser()
        self.cf.read(conf_path)

    def get_path(self, param):
        loc_path = self.cf.get('Paths', param)
        root_path = Path(__file__).resolve().parent.parent.as_posix()
        return root_path+loc_path


if __name__ == "__main__":
    test = ReadConfig()
    t = test.get_path('train_raw')
    print(t)


        



