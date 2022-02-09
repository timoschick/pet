import os
import sys
import random


def main():
    random.seed(42)
    file = sys.argv[1]
    sample_count = int(sys.argv[2])

    with open(file) as f:
        lines = f.readlines()
        
    header, data = lines[0], lines[1:]
    random.shuffle(data)

    dirname, basename = os.path.split(file)
    prefix, ext = basename.split('.')
    write_name = os.path.join(dirname, f"{prefix}_{sample_count}.{ext}")

    with open(write_name, "w") as f:
        f.writelines([header] + data[:sample_count])


if __name__ == "__main__":
    main()
