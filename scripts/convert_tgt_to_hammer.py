import argparse

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument('--source-path', '-sp', type=str, required=True)
    args.add_argument('--target-path', '-tp', type=str, required=True)
    parser = args.parse_args()

    with open(parser.target_path, 'w') as fout, open(parser.source_path, 'r') as fin:
        for line in fin.readlines():
            line = line.strip()
            if "metis" in line or "meson" in line or "smt" in line:
                line = "sledgehammer"
            fout.write(line + "\n")
