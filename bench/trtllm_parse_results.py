import os
import json
import argparse

from parse_common import (parse_run_params, run_params_to_string,
                          parse_filename, write_results)


def parse_results(results_dir, run_params):
    filenames = os.listdir(results_dir)

    files_dict = {}
    for i, filename in enumerate(filenames):
        if filename.find("__batch_") == -1:
            continue

        file_params = parse_filename(filename)
        files_dict[file_params["batch"]] = (filename, file_params)

    results_dict = {}
    for i, item in enumerate(sorted(files_dict.items())):
        batch_size, file_data = item
        filename, file_params = file_data

        print("[{}] file = {}".format(i + 1, filename))

        assert run_params["isl"] == int(file_params["isl"])
        assert run_params["osl"] == int(file_params["osl"])
        assert run_params["tp"] == int(file_params["TP"])

        with open(os.path.join(args.results_dir, filename), 'r') as f:
            data = json.load(f)
            total_tokens_per_sec = data['performance'][
                'system_total_throughput_tok_s']
            output_tokens_per_sec = data['performance'][
                'system_output_throughput_tok_s']

        print("    batch_size            = {}".format(batch_size))
        print("    total_tokens_per_sec  = {}".format(total_tokens_per_sec))
        print("    output_tokens_per_sec = {}".format(output_tokens_per_sec))
        results_dict[batch_size] = (total_tokens_per_sec,
                                    output_tokens_per_sec)

    return results_dict


def main(args):
    print("Get run params:")
    run_params = parse_run_params(args.results_dir, "trtllm")
    print(run_params_to_string(run_params))

    print("Parsing results from {}".format(args.results_dir))
    results_dict = parse_results(args.results_dir, run_params)

    print("Writing results to {}".format(args.output_file))
    write_results(args.output_file, run_params, results_dict)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=str, help="Path to results dir")
    parser.add_argument("--output-file", type=str, help="Path to output file")

    args = parser.parse_args()
    main(args)
