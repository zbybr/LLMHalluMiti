"""Model-specific command-line entry point; implementation is shared."""

from _experiment import run


if __name__ == "__main__":
    run("qa.drhall", __file__)
