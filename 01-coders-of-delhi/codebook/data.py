"""Loading, saving and indexing the CodeBook JSON dump."""
import json


def load_data(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_data(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def friend_sets(data):
    """user id -> set of friend ids."""
    return {u["id"]: set(u["friends"]) for u in data["users"]}


def like_sets(data):
    """user id -> set of liked page ids."""
    return {u["id"]: set(u["liked_pages"]) for u in data["users"]}


def names(data):
    return {u["id"]: u["name"] for u in data["users"]}


def page_names(data):
    return {p["id"]: p["name"] for p in data["pages"]}


def describe(data):
    """Readable listing of users, their friends and liked pages."""
    users, pages = names(data), page_names(data)
    lines = ["Users"]
    for u in data["users"]:
        friends = ", ".join(users.get(f, f"#{f}") for f in u["friends"]) or "-"
        liked = ", ".join(pages.get(p, f"#{p}") for p in u["liked_pages"]) or "-"
        lines.append(f"  {u['name']} (id {u['id']}): friends [{friends}] | likes [{liked}]")
    lines.append("Pages")
    lines += [f"  {p['id']}: {p['name']}" for p in data["pages"]]
    return "\n".join(lines)
