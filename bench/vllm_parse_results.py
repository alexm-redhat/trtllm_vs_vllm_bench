import os
import json
import argparse

from parse_common import (parse_run_params, run_params_to_string,
                          parse_filename, write_results)


def parse_results(results_dir, run_params):
    filenames = os.listdir(results_dir)

    isl = run_params["isl"]
    osl = run_params["osl"]

    results_dict = {}
    for i, filename in enumerate(sorted(filenames)):
        if filename.find("__batch_") == -1:
            continue

        print("[{}] file = {}".format(i + 1, filename))

        file_params = parse_filename(filename)
        assert isl == int(file_params["isl"])
        assert osl == int(file_params["osl"])
        assert run_params["tp"] == int(file_params["TP"])

        batch_size = file_params["batch"]
        num_prompts = file_params["num-prompts"]

        with open(os.path.join(args.results_dir, filename), 'r') as f:
            data = json.load(f)
            total_tokens_per_sec = data['tokens_per_second']

            elapsed_time = data['elapsed_time']
            total_num_tokens = data['total_num_tokens']
            
            expected_total_num_tokens = (isl + osl) * num_prompts
            assert expected_total_num_tokens == total_num_tokens, (
                "expected_total_num_tokens = {} total_num_tokens = {}".format(
                    expected_total_num_tokens, total_num_tokens))
            
            output_tokens_per_sec = (osl * num_prompts) / elapsed_time

        print("    batch_size            = {}".format(batch_size))
        print("    total_tokens_per_sec  = {}".format(total_tokens_per_sec))
        print("    output_tokens_per_sec = {}".format(output_tokens_per_sec))
        results_dict[batch_size] = (total_tokens_per_sec,
                                    output_tokens_per_sec)

    return results_dict


def main(args):
    print("Get run params:")
    run_params = parse_run_params(args.results_dir, "vllm")
    print(run_params_to_string(run_params))

    results_dict = parse_results(args.results_dir, run_params)

    print("Writing results to {}".format(args.output_file))
    write_results(args.output_file, run_params, results_dict)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=str, help="Path to results dir")
    parser.add_argument("--output-file", type=str, help="Path to output file")

    args = parser.parse_args()
    main(args)
