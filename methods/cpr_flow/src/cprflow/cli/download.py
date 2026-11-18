from cprflow.cli.common import config_parser, print_json, resolved
from cprflow.data.sources import download_authorized


def main():
    args = config_parser(
        "Download only the three authorized corpora at the locked revision"
    ).parse_args()
    print_json({"destination": str(download_authorized(resolved(args)))})


if __name__ == "__main__":
    main()
