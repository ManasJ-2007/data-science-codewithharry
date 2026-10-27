"""CodeBook command line: clean the data dump and show recommendations.

Usage:
    python main.py                       # clean data/codebook_data.json, show suggestions for every user
    python main.py --user 1 --k 3        # one user
    python main.py --people-method adamic_adar --page-method hybrid
"""
import argparse
import os

from codebook import clean_data, load_data, pages_you_might_like, people_you_may_know, save_data
from codebook.data import describe, names, page_names
from codebook.recommend import PAGE_METHODS, PEOPLE_METHODS

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=os.path.join(HERE, "data", "codebook_data.json"))
    ap.add_argument("--user", type=int, help="user id (default: all users)")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--people-method", default="mutual", choices=PEOPLE_METHODS)
    ap.add_argument("--page-method", default="overlap", choices=PAGE_METHODS)
    args = ap.parse_args()

    data, report = clean_data(load_data(args.data))
    out = os.path.join(os.path.dirname(args.data), "cleaned_codebook_data.json")
    save_data(data, out)
    print("Cleaning report:", ", ".join(f"{k}={v}" for k, v in report.items() if v))
    print(f"Saved cleaned data to {os.path.relpath(out, HERE)}\n")
    print(describe(data), "\n")

    users, pages = names(data), page_names(data)
    ids = [args.user] if args.user else [u["id"] for u in data["users"]]
    for uid in ids:
        if uid not in users:
            print(f"User {uid} not found (removed during cleaning or never existed)")
            continue
        people = people_you_may_know(uid, data, args.people_method, args.k)
        liked = pages_you_might_like(uid, data, args.page_method, args.k)
        print(f"{users[uid]} (id {uid})")
        print("  People you may know:", ", ".join(users[p] for p in people) or "-")
        print("  Pages you might like:", ", ".join(pages[p] for p in liked) or "-")


if __name__ == "__main__":
    main()
