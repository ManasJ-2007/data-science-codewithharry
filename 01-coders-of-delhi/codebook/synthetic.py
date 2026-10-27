"""Synthetic CodeBook network for evaluating the recommenders at scale.

Each user belongs to one interest community (Python, Web Dev, ...). Friendships come from a
stochastic block model (dense inside a community, sparse across) followed by triadic closure
("friends of friends become friends"), which is what makes mutual-friend signals realistic.
Likes are drawn mostly from the user's community pages with Zipf-like page popularity, plus
some likes copied from friends (social influence). Optionally injects dirty records to
exercise the cleaning step.
"""
import random

COMMUNITIES = ["Python", "Web Dev", "Machine Learning", "DSA & CP", "DevOps",
               "Android", "Blockchain", "Game Dev"]
FIRST_NAMES = ["Aarav", "Vivaan", "Aditya", "Vihaan", "Arjun", "Sai", "Reyansh", "Krishna", "Ishaan",
               "Rohan", "Ananya", "Diya", "Priya", "Isha", "Kavya", "Meera", "Riya", "Sara", "Tanvi",
               "Neha", "Amit", "Rahul", "Karan", "Nikhil", "Pooja", "Sneha", "Manas", "Yash", "Aditi", "Varun"]
PAGE_SUFFIXES = ["Hub", "Daily", "Community", "Beginners", "Pros", "Projects", "Interview Prep", "News"]


def generate(n_users=1000, avg_friends=12, p_cross=0.08, closure_steps=None,
             likes_per_user=6, p_copy_friend=0.3, dirty_fraction=0.0, seed=0):
    rng = random.Random(seed)
    pages = []
    for c, topic in enumerate(COMMUNITIES):
        for s, suffix in enumerate(PAGE_SUFFIXES):
            pages.append({"id": 101 + c * len(PAGE_SUFFIXES) + s, "name": f"{topic} {suffix}", "community": c})
    by_comm = {c: [p["id"] for p in pages if p["community"] == c] for c in range(len(COMMUNITIES))}
    zipf = [1.0 / (r + 1) for r in range(len(PAGE_SUFFIXES))]

    comm = {uid: rng.randrange(len(COMMUNITIES)) for uid in range(1, n_users + 1)}
    members = {c: [u for u, cu in comm.items() if cu == c] for c in range(len(COMMUNITIES))}
    friends = {u: set() for u in comm}

    def link(a, b):
        if a != b:
            friends[a].add(b)
            friends[b].add(a)

    target_edges = n_users * avg_friends // 2
    sbm_edges = int(target_edges * 0.6)
    while sum(len(f) for f in friends.values()) // 2 < sbm_edges:
        a = rng.randrange(1, n_users + 1)
        pool = list(comm) if rng.random() < p_cross else members[comm[a]]
        link(a, rng.choice(pool))
    steps = closure_steps if closure_steps is not None else 20 * n_users
    for _ in range(steps):
        if sum(len(f) for f in friends.values()) // 2 >= target_edges:
            break
        z = rng.randrange(1, n_users + 1)
        if len(friends[z]) >= 2:
            a, b = rng.sample(sorted(friends[z]), 2)
            link(a, b)

    likes = {u: set() for u in comm}
    for u in comm:
        n = max(1, int(rng.gauss(likes_per_user, 2)))
        for _ in range(n):
            c = comm[u] if rng.random() < 0.8 else rng.randrange(len(COMMUNITIES))
            likes[u].add(rng.choices(by_comm[c], weights=zipf)[0])
    for u in comm:
        for f in friends[u]:
            if likes[f] and rng.random() < p_copy_friend / max(len(friends[u]), 1) * 3:
                likes[u].add(rng.choice(sorted(likes[f])))

    users = [{"id": u, "name": f"{rng.choice(FIRST_NAMES)} {u}", "friends": sorted(friends[u]),
              "liked_pages": sorted(likes[u])} for u in comm]
    pages_out = [{"id": p["id"], "name": p["name"]} for p in pages]

    if dirty_fraction:
        k = int(n_users * dirty_fraction)
        for u in rng.sample(users, k):
            kind = rng.randrange(5)
            if kind == 0:
                u["name"] = "  "
            elif kind == 1 and u["friends"]:
                u["friends"].append(u["friends"][0])
            elif kind == 2:
                u["friends"].append(u["id"])
            elif kind == 3:
                u["liked_pages"].append(999)
            else:
                u["friends"].append(n_users + 500)
        for _ in range(k // 5):
            users.append({"id": n_users + 1 + len(users), "name": "Ghost", "friends": [], "liked_pages": []})
        pages_out.append({"id": pages_out[0]["id"], "name": pages_out[0]["name"] + " (old)"})

    return {"users": users, "pages": pages_out}, comm
