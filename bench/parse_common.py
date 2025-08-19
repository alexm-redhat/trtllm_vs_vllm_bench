def parse_run_params(results_dir, expected_framework):
    parts = results_dir.strip("/").split("/")
    framework, _, isl, _, osl, _, tp = parts[-1].split("_")
    assert framework == expected_framework
    isl = int(isl)
    osl = int(osl)
    tp = int(tp)
    model_name = parts[-2]

    return {
        "framework": framework,
        "isl": isl,
        "osl": osl,
        "tp": tp,
        "model_name": model_name
    }


def run_params_to_string(run_params):
    s1 = "    FRAMEWORK = {}".format(run_params["framework"])
    s2 = "    MODEL     = {}".format(run_params["model_name"])
    s3 = "    ISL       = {}".format(run_params["isl"])
    s4 = "    OSL       = {}".format(run_params["osl"])
    s5 = "    TP        = {}".format(run_params["tp"])
    return "\n".join([s1, s2, s3, s4, s5])


def parse_filename(filename):
    parts = filename.strip(".json").split("__")
    model = parts[0]
    isl = parts[1]
    osl = parts[2]
    tp = parts[3]
    batch_size = parts[4]

    res_dict = {}
    expected_names = ["isl", "osl", "TP", "num-prompts", "batch"]
    for i, part in enumerate(parts[2:]):
        name, val = part.split("_")
        assert name == expected_names[
            i], "name = {}, expected_name = {}".format(name, expected_names[i])
        res_dict[name] = int(val)

    return res_dict


def write_results(output_file, run_params, results_dict):
    with open(output_file, 'w') as f:
        f.write("{}\n".format(run_params_to_string(run_params)))
        f.write("batch_size | total_tokens_per_sec | output_tokens_per_sec\n")
        for batch_size, results in sorted(results_dict.items()):
            total_tokens_per_sec, output_tokens_per_sec = results
            f.write("{} | {} | {}\n".format(batch_size, total_tokens_per_sec,
                                            output_tokens_per_sec))
