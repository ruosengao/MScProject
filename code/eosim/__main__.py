#!/usr/bin/env python3

import argparse
import datetime
import json
import time

from tabulate import tabulate

from . import oop
from . import procedural as pro

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["oop", "pro"],
        help="mode of simulation")
    parser.add_argument("simulator", choices=["exit-time", "occup-time"],
        help="simulator")
    parser.add_argument("config", help="configuration file")
    args = parser.parse_args()

    with open(args.config) as file:
        data = json.load(file)

    # parse domains
    domain_names = ["OpenBall", "OpenAnnulus"]
    
    def domain_parser(domain):
        name, *para = domain
        if name not in domain_names:
            raise RuntimeError(f"{name} is not a valid domain")
        if args.mode == "oop":
            if name == "OpenBall":
                c, r = para
                return oop.OpenBall(c, r)
            else: # name == "OpenAnnulus"
                c, r1, r2 = para
                return oop.OpenAnnulus(c, r1, r2)
        else: # args.mode == "pro"
            return tuple(domain)

    if args.simulator == "exit-time":
        domain = domain_parser(data["domain"])
    else: # args.simulator == "occup-time"
        domain_d = domain_parser(data["domain_d"])
        domain_v = domain_parser(data["domain_v"])

    # estimation
    t0 = time.perf_counter()
    if (args.mode, args.simulator) == ("oop", "exit-time"):
        est = oop.ExitTimeSimulator(
            domain, data["max_t"], data["dt"], data["dx"], data["n"]
        ).run()
    elif (args.mode, args.simulator) == ("oop", "occup-time"):
        est = oop.OccupationTimeSimulator(
            domain_d, domain_v, data["max_t"], data["dt"], data["dx"], data["n"]
        ).run()
    elif (args.mode, args.simulator) == ("pro", "exit-time"):
        est = pro.simulate_max_expected_exit_time(
            domain, data["max_t"], data["dt"], data["dx"], data["n"])
    else: # (args.mode, args.simulator) == ("pro", "occup-time")
        est = pro.simulate_min_expected_occupation_time(
            domain_d, domain_v, data["max_t"], data["dt"], data["dx"], data["n"]
        )
    t1 = time.perf_counter()

    # print message
    msg = [
        ["Grid", f'max_t={data["max_t"]}, dt={data["dt"]}, dx={data["dx"]}'],
        ["No. Samples", f'{data["n"]} per gridpoint'],
        ["Estimate", est],
        ["Performance", datetime.timedelta(seconds=t1-t0)]
    ]
    if args.simulator == "exit-time":
        msg.insert(0, [f"ExitTimeSimulator ({args.mode})"])
        msg.insert(1, ["Domain", domain])
    else: # args.simulator == "occup-time"
        msg.insert(0, [f"OccupationTimeSimulator ({args.mode})"])
        msg.insert(1, ["Domain D, V", f"{domain_d},\n{domain_v}"])
    print(tabulate(msg, headers="firstrow", tablefmt="fancy_grid"))
