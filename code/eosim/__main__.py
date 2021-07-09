#!/usr/bin/env python3

import argparse
import datetime
import json
import time

from tabulate import tabulate


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["oop", "procedural"],
        help="mode of simulation")
    parser.add_argument("simulator", choices=["exit-time", "occup-time"],
        help="simulator")
    parser.add_argument("config", help="configuration file")
    args = parser.parse_args()

    with open(args.config) as file:
        data = json.load(file)

    # compute result
    t0 = time.perf_counter()
    if args.mode == "oop":
        from . import oop
        result = oop.main(args.simulator, **data)
    else: # args.mode == "procedural"
        from . import procedural
        result = procedural.main(args.simulator, **data)
    t1 = time.perf_counter()

    # print message
    msg = [
        ["Grid", f'max_t={data["max_t"]}, dt={data["dt"]}, dx={data["dx"]}'],
        ["No. Samples", f'{data["n"]} per gridpoint'],
        ["Estimate", result],
        ["Performance", datetime.timedelta(seconds=t1-t0)]
    ]
    if args.simulator == "exit-time":
        msg.insert(0, [f"ExitTimeSimulator ({args.mode})"])
        msg.insert(1, ["Domain", data["domain"]])
    else: # args.simulator == "occup-time"
        msg.insert(0, [f"OccupationTimeSimulator ({args.mode})"])
        msg.insert(1, ["Domain D, V",
            f'{data["domain_d"]},\n{data["domain_v"]}'])
    print(tabulate(msg, headers="firstrow", tablefmt="fancy_grid"))
