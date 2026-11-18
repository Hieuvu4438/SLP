from cprflow.cli.common import config_parser, print_json, resolved
from cprflow.data.sources import audit_sources


def main():
    parser = config_parser("Audit sources, environment and pinned revisions")
    parser.add_argument("--resolve-lock", action="store_true")
    args = parser.parse_args()
    print_json(audit_sources(resolved(args), args.resolve_lock))


if __name__ == "__main__":
    main()
