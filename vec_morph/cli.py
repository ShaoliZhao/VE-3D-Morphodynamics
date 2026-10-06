from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .core import forecast_log_scale, interpolate_population, relational_features, rms_radius


def describe(args):
    data = np.load(args.input)
    coords = data["coords"]
    print(
        json.dumps(
            {
                "cells": len(coords),
                "rms_radius": rms_radius(coords),
                "descriptor_dimensions": relational_features(coords).shape[1],
            },
            indent=2,
        )
    )


def interpolate(args):
    data = np.load(args.input)
    result = interpolate_population(
        data["coords_a"],
        data["coords_b"],
        data["expr_a"],
        data["expr_b"],
        args.alpha,
        expression_weight=args.expression_weight,
        geometry_weight=args.geometry_weight,
        k=args.neighbors,
    )
    np.savez_compressed(args.output, **result, alpha=np.array(args.alpha))
    print(
        json.dumps(
            {
                "output": str(Path(args.output).resolve()),
                "cells": len(result["coords"]),
                "predicted_scale": result["predicted_scale"],
            },
            indent=2,
        )
    )


def scale(args):
    data = np.load(args.input)
    print(
        json.dumps(
            {
                "target_time": args.target_time,
                "predicted_scale": forecast_log_scale(data["times"], data["scales"], args.target_time),
            },
            indent=2,
        )
    )


def main():
    parser = argparse.ArgumentParser(description="Frame-aware Task 2 morphogenesis baselines")
    sub = parser.add_subparsers(dest="command", required=True)
    command = sub.add_parser("describe")
    command.add_argument("input")
    command.set_defaults(func=describe)

    command = sub.add_parser("interpolate")
    command.add_argument("input")
    command.add_argument("output")
    command.add_argument("--alpha", type=float, required=True)
    command.add_argument("--expression-weight", type=float, default=0.5)
    command.add_argument("--geometry-weight", type=float, default=0.5)
    command.add_argument("--neighbors", type=int, default=8)
    command.set_defaults(func=interpolate)

    command = sub.add_parser("forecast-scale")
    command.add_argument("input")
    command.add_argument("--target-time", type=float, required=True)
    command.set_defaults(func=scale)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
