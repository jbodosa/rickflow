# -*- coding: utf-8 -*-

"""Console script for rickflow."""
import sys
import click


@click.group()
def main(args=None):
    """Console script for rickflow."""
    click.echo("Rickflow: a python package to facilitate running jobs in OpenMM using CHARMM defaults.")
    return 0


@main.command()
@click.option()
def create():
    pass


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
