import os
import yaml
import contextlib
# from contextlib import nullcontext


def recursive_update(ori_dict, new_dict):
    for k in new_dict:
        if k not in ori_dict:
            ori_dict[k] = new_dict[k]
        elif not isinstance(ori_dict[k], dict):
            ori_dict[k] = new_dict[k]
        else:
            ori_value, new_value = ori_dict[k], new_dict[k]
            ori_dict[k] = recursive_update(ori_value, new_value)
    return ori_dict


def load_yaml(fn):
    with open(fn) as f:
        config = yaml.load(f)
    if "_base_" in config:
        base_path = config["_base_"]
        base_config = load_yaml(base_path)
        config = recursive_update(ori_dict=base_config, new_dict=config)
        del config["_base_"]
    return config


def flatten_dict(dct, parent_keyname=None):
    flattened = {}
    for k, value in dct.items():
        key = f"{parent_keyname}.{k}" if parent_keyname is not None else k
        if isinstance(value, dict):
            flattened.update(flatten_dict(value, parent_keyname=key))
        else:
            flattened[key] = value
    return flattened


def record_commit_id(fn):
    with open(fn, "a") as f:
        f.write("\n" f"# current commit id: {find_current_commit_id()}")


def find_current_commit_id():
    return os.popen("git log | head -n 1").readline().strip().split()[1]