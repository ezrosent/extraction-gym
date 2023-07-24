#!/usr/bin/env python3
import glob
import json
import statistics
import sys


def load_jsons(files):
    js = []
    for file in files:
        try:
            with open(file) as f:
                j = json.load(f)
                j["json_path"] = file
                js.append(j)
        except Exception as e:
            print(f"Error loading {file}")
            raise e
    return js


def process(js):
    by_name = {}
    for j in js:
        n, e = j["name"], j["extractor"]
        by_name.setdefault(n, {})[e] = j

    print(f"name\textractor\ttree\tdag\tmicros")

    for name, d in by_name.items():
        try:
            for e in d:
                tree = d[e]["tree"]
                dag = d[e]["dag"]
                micros = d[e]["micros"]
                print(f"{name}\t{e}\t{tree}\t{dag}\t{micros}")
        except Exception as e:
            print(f"Error processing {name}")
            # raise e


if __name__ == "__main__":
    print()
    print(" ------------------------ ")
    print(" ------- raw data ------- ")
    print(" ------------------------ ")
    print()
    files = sys.argv[1:] or glob.glob("output/**/*.json", recursive=True)
    js = load_jsons(files)
    print(f"Loaded {len(js)} jsons.")
    process(js)
