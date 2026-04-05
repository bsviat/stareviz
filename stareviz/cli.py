import argparse, webbrowser
from threading import Timer

from stareviz.app import build_server


def main():
    p = argparse.ArgumentParser("STAREVOL visualizer")
    p.add_argument("roots", nargs="*", help="Root directories with models (optional if set in stareviz.yml)")
    p.add_argument("--port", type=int, default=8050)
    args = p.parse_args()

    roots = args.roots
    if not roots:
        from stareviz.viz_config import cfg
        roots = cfg.default_roots
        if not roots:
            p.error(
                "No model directories specified. "
                "Pass them as arguments or set 'default_roots' in stareviz.yml."
            )

    server, url = build_server(roots, port=args.port)
    print("Welcome to STAREVIZ, visualization tool for the STAREVOL data.")
    print("Developed by Sviatoslav Borisov.")
    print("Please send suggestions for improvement and bug reports to borisov.sviat@gmail.com")
    print("Press CTRL+C to quit")
    Timer(0.1, lambda: webbrowser.open(url)).start()
    server.run(host="127.0.0.1", port=args.port, debug=False)