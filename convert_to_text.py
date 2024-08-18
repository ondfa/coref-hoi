import sys

import udapi_io
from run import Runner

if __name__ == '__main__':
    config_name, gpu_id = sys.argv[1], int(sys.argv[2])
    runner = Runner(config_name, gpu_id)
    udapi_io.convert_all_to_text(runner.config)