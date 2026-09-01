"""Command line interface.

Four subcommands::

    alpinemet datasets                     # list the registered products
    alpinemet check   --config <file>      # validate a configuration
    alpinemet run     --config <file>      # load, compute indicators, write results
    alpinemet figures --config <file>      # regenerate manuscript figures

``check`` exists because the alternative -- discovering a typo after a
multi-hour load -- is the most avoidable failure mode in this workflow. It
validates the configuration, resolves every dataset path against the data root
and reports which inputs are missing, without opening a single file.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

from alpinemet import __version__

LOGGER = logging.getLogger("alpinemet")

DEFAULT_PATHS_FILE = Path("configs/paths.yaml")


def _configure_logging(verbosity: int) -> None:
    level = logging.WARNING if verbosity == 0 else logging.INFO if verbosity == 1 else logging.DEBUG
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser."""
    parser = argparse.ArgumentParser(
        prog="alpinemet",
        description=(
            "Multi-scale weather and climate data evaluation for Alpine renewable energy"
        ),
    )
    parser.add_argument("--version", action="version", version=f"alpinemet {__version__}")
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="increase log verbosity; repeat for debug output",
    )

    subparsers = parser.add_subparsers(dest="command", metavar="<command>")

    subparsers.add_parser("datasets", help="list the registered products")

    for name, help_text in (
        ("check", "validate a configuration and report missing inputs"),
        ("run", "run an analysis"),
        ("figures", "regenerate manuscript figures"),
    ):
        sub = subparsers.add_parser(name, help=help_text)
        sub.add_argument("-c", "--config", required=True, type=Path, help="analysis YAML")
        sub.add_argument(
            "-p",
            "--paths",
            type=Path,
            default=DEFAULT_PATHS_FILE,
            help=f"machine-specific paths file (default: {DEFAULT_PATHS_FILE})",
        )
        if name != "check":
            sub.add_argument(
                "-o", "--output-dir", type=Path, default=None, help="override the output directory"
            )

    return parser


def _command_datasets() -> int:
    from alpinemet.io.datasets import DATASETS

    header = f"{'key':<14}{'resolution':>12}{'step':>7}  {'gust':^6}{'100m':^6}  name"
    print(header)
    print("-" * len(header))
    for key in sorted(DATASETS):
        spec = DATASETS[key]
        print(
            f"{spec.key:<14}{spec.resolution_km:>9.1f} km{spec.timestep_hours:>5.0f} h"
            f"  {'yes' if spec.has_native_gust else 'no':^6}"
            f"{'yes' if spec.has_100m_wind else 'no':^6}  {spec.long_name}"
        )
    return 0


def _command_check(config_path: Path, paths_path: Path) -> int:
    from alpinemet.config import load_config, load_paths

    config = load_config(config_path)
    paths = load_paths(paths_path)

    print(f"Configuration : {config.name}")
    if config.description:
        print(f"Description   : {config.description}")
    print(f"Domain        : {config.domain.name} "
          f"({config.domain.lon_min}-{config.domain.lon_max} degE, "
          f"{config.domain.lat_min}-{config.domain.lat_max} degN)")
    print(f"Period        : {config.start or 'start of record'} to {config.end or 'end of record'}")
    print(f"Data root     : {paths.data_root}")
    print()

    missing = 0
    for entry in config.datasets:
        resolved = paths.resolve(entry.paths)
        if any(character in str(resolved) for character in "*?["):
            from glob import glob

            matches = glob(str(resolved))
            status = f"{len(matches)} file(s)" if matches else "NO MATCH"
            if not matches:
                missing += 1
        else:
            status = "found" if resolved.exists() else "MISSING"
            if not resolved.exists():
                missing += 1
        print(f"  {entry.key:<14} {status:<12} {resolved}")

    print()
    if missing:
        print(f"{missing} of {len(config.datasets)} inputs could not be found.")
        return 1
    print(f"All {len(config.datasets)} inputs found. Configuration is valid.")
    return 0


def _command_run(config_path: Path, paths_path: Path, output_dir: Path | None) -> int:
    from alpinemet.config import load_config, load_paths
    from alpinemet.io import open_product

    config = load_config(config_path)
    paths = load_paths(paths_path)
    destination = output_dir or (paths.output_root / config.name)
    destination.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Running %s, writing to %s", config.name, destination)

    for entry in config.datasets:
        resolved = paths.resolve(entry.paths)
        LOGGER.info("Loading %s from %s", entry.key, resolved)
        dataset = open_product(
            str(resolved),
            entry.key,
            domain=config.domain,
            start=config.start,
            end=config.end,
        )
        LOGGER.info(
            "  %s: %s variables, %s time steps",
            entry.key,
            len(dataset.data_vars),
            dataset.sizes.get("time", "n/a"),
        )
        target = destination / f"{entry.key}_standardised.nc"
        dataset.to_netcdf(target)
        print(f"wrote {target}")

    return 0


def _command_figures(config_path: Path, paths_path: Path, output_dir: Path | None) -> int:
    print(
        "Figure regeneration is not wired up yet. The manuscript figures are\n"
        "produced from the derived fields published on Zenodo; see the README\n"
        "section 'Reproducing the manuscript figures'."
    )
    return 2


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point.

    Returns
    -------
    int
        Process exit status: 0 on success, 1 on a user-facing failure such as
        a missing input or an invalid configuration, 2 for an unimplemented
        command.
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    _configure_logging(getattr(args, "verbose", 0))

    if args.command is None:
        parser.print_help()
        return 1

    try:
        if args.command == "datasets":
            return _command_datasets()
        if args.command == "check":
            return _command_check(args.config, args.paths)
        if args.command == "run":
            return _command_run(args.config, args.paths, args.output_dir)
        if args.command == "figures":
            return _command_figures(args.config, args.paths, args.output_dir)
    except (FileNotFoundError, ValueError) as exc:
        # Configuration and input problems are the user's to fix; a traceback
        # would only bury the message.
        print(f"error: {exc}", file=sys.stderr)
        return 1

    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
