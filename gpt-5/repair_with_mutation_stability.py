"""GPT-5 entry point for the shared full-pipeline stability experiment."""

from _experiment import run


if __name__ == "__main__":
    run("common.stability_entry", __file__)
