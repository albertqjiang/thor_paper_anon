import argparse
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Distribution of proof length in the number of steps.')
    parser.add_argument('--seq2seq-dir', '-sd', type=str)
    args = parser.parse_args()

    train_path = os.path.join(args.seq2seq_dir, "train.src")
    val_path = os.path.join(args.seq2seq_dir, "val.src")
    test_path = os.path.join(args.seq2seq_dir, "test.src")

    theorem_lengths = []
    for file_path in [train_path, val_path, test_path]:
        with open(file_path) as fh:
            current_length = -1
            for line in fh.readlines():
                if line.startswith("<ISA_LAST_1> lemma") or line.startswith("<ISA_LAST_1> theorem"):
                    if current_length > 0:
                        theorem_lengths.append(current_length)
                    current_length = 1
                else:
                    current_length += 1

    theorem_distribution = {}
    for length in theorem_lengths:
        if length not in theorem_distribution:
            theorem_distribution[length] = 0
        theorem_distribution[length] += 1

    total = 0
    for length in sorted(theorem_distribution.keys()):
        print(f"{length}: {theorem_distribution[length]}")
        total += length
    print(f"Total: {total}")
