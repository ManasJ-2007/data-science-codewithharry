"""Cleaning a messy CodeBook dump.

Rules, applied in order:
1. drop users with a missing id or a blank name; keep the first record of a duplicated user id
2. keep the first record of a duplicated page id
3. friend lists: drop self-references, unknown user ids and duplicates
4. liked pages: drop unknown page ids and duplicates
5. make friendship symmetric (if A lists B, B also lists A)
6. drop inactive users (no friends and no liked pages)
"""
import copy
from collections import Counter


def clean_data(raw):
    """Return (cleaned_data, report) without modifying `raw`."""
    data = copy.deepcopy(raw)
    report = Counter()

    users, seen = [], set()
    for u in data.get("users", []):
        uid, name = u.get("id"), (u.get("name") or "").strip()
        if uid is None or not name:
            report["users_missing_id_or_name"] += 1
            continue
        if uid in seen:
            report["duplicate_user_ids"] += 1
            continue
        seen.add(uid)
        users.append({"id": uid, "name": name,
                      "friends": list(u.get("friends") or []),
                      "liked_pages": list(u.get("liked_pages") or [])})

    pages, page_ids = [], set()
    for p in data.get("pages", []):
        if p.get("id") in page_ids:
            report["duplicate_page_ids"] += 1
            continue
        page_ids.add(p.get("id"))
        pages.append({"id": p["id"], "name": (p.get("name") or "").strip()})

    user_ids = {u["id"] for u in users}
    for u in users:
        friends = []
        for f in u["friends"]:
            if f == u["id"]:
                report["self_friendships"] += 1
            elif f not in user_ids:
                report["unknown_friend_ids"] += 1
            elif f in friends:
                report["duplicate_friend_entries"] += 1
            else:
                friends.append(f)
        u["friends"] = friends

        liked = []
        for p in u["liked_pages"]:
            if p not in page_ids:
                report["unknown_page_ids"] += 1
            elif p in liked:
                report["duplicate_likes"] += 1
            else:
                liked.append(p)
        u["liked_pages"] = liked

    by_id = {u["id"]: u for u in users}
    for u in users:
        for f in list(u["friends"]):
            if u["id"] not in by_id[f]["friends"]:
                by_id[f]["friends"].append(u["id"])
                report["one_sided_friendships_fixed"] += 1

    active = [u for u in users if u["friends"] or u["liked_pages"]]
    report["inactive_users_removed"] = len(users) - len(active)
    for u in active:
        u["friends"].sort()
        u["liked_pages"].sort()

    return {"users": active, "pages": pages}, dict(report)
