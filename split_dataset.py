import os
from pathlib import Path
import argparse
import shutil
from concurrent.futures import ThreadPoolExecutor
import time


def chunks(xs, n):
    n = max(1, n)
    return (xs[i:i+n] for i in range(0, len(xs), n))

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True,
                help="path to input directory")
ap.add_argument('-s', '--split', required=True, type=int,
                help='number of splits of the dataset')
args = vars(ap.parse_args())


# get filenames
fnames = [fname for fname in os.listdir(args['input']) if fname.endswith('.fits')]
num_files = int(len(fnames) / args['split'])

# check if path exists and files already downloaded
for i in range(1, args['split'] + 1):
    # create output path
    Path(os.path.join(args['input'], 'split', str(i))).mkdir(parents=True, exist_ok=True)


def copy(chunk_id, chunk):
    for fname in chunk:
        os.rename(os.path.join(args['input'], fname), os.path.join(args['input'], 'split', str(chunk_id + 1), fname))

if __name__ == "__main__":
    with ThreadPoolExecutor() as executor:
        
        print(f"started at {time.strftime('%X')}")
        
        # launch downloads over cpus
        futures = {executor.submit(copy, chunk_id, chunk) for chunk_id, chunk in enumerate(chunks(fnames, num_files))}
    
    print(f"finished at {time.strftime('%X')}")