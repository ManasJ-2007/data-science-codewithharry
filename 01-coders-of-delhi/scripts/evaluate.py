"""Offline evaluation of the recommenders on synthetic CodeBook networks (3 seeds).

People You May Know: hide 10% of friendships, then ask whether the hidden friend shows up in
the user's top-k suggestions (Hit@k) and at what rank (MRR).
Pages You Might Like: hide one liked page per user (users with >= 3 likes) and do the same.
Cleaning: inject dirty records into 10% of users and check every integrity rule afterwards.
Speed: time to compute recommendations for every user.

Usage:  python scripts/evaluate.py      ->  results/evaluation.json, results/evaluation.md
"""
import json
import os
import random
import statistics
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from codebook.cleaning import clean_data  # noqa: E402
from codebook.data import friend_sets, like_sets  # noqa: E402
from codebook.recommend import PAGE_METHODS, PEOPLE_METHODS, score_pages, score_people  # noqa: E402
from codebook.synthetic import generate  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "..", "results")
SEEDS = (0, 1, 2)
N_USERS = 1000


def ranks_of(scores, target):
    """1-based rank of target under deterministic ordering, or None if unscored."""
    if target not in scores or scores[target] <= 0:
        return None
    s = scores[target]
    return 1 + sum(1 for k, v in scores.items() if v > s or (v == s and k < target))


def summarise(ranks, ks=(1, 5, 10)):
    n = len(ranks)
    out = {f"hit@{k}": sum(1 for r in ranks if r is not None and r <= k) / n for k in ks}
    out["mrr"] = sum(1 / r for r in ranks if r is not None) / n
    return out


def random_baseline(candidate_counts, ks=(1, 5, 10)):
    """Expected Hit@k and MRR for a uniformly random ordering of each user's candidates."""
    out = {f"hit@{k}": statistics.mean(min(k, c) / c for c in candidate_counts) for k in ks}
    out["mrr"] = statistics.mean(sum(1 / r for r in range(1, c + 1)) / c for c in candidate_counts)
    return out


def eval_people(data, seed):
    rng = random.Random(seed + 100)
    friends = friend_sets(data)
    edges = sorted({(min(a, b), max(a, b)) for a, fs in friends.items() for b in fs})
    rng.shuffle(edges)
    hidden = []
    for a, b in edges:
        if len(hidden) >= len(edges) // 10:
            break
        if len(friends[a]) > 1 and len(friends[b]) > 1:
            friends[a].discard(b)
            friends[b].discard(a)
            hidden.append((a, b) if rng.random() < 0.5 else (b, a))
    res = {}
    for m in PEOPLE_METHODS:
        cache, ranks = {}, []
        for u, v in hidden:
            if u not in cache:
                cache[u] = score_people(u, friends, m)
            ranks.append(ranks_of(cache[u], v))
        res[m] = summarise(ranks)
    n = len(friends)
    res["random"] = random_baseline([n - 1 - len(friends[u]) for u, _ in hidden])
    return res, len(hidden)


def eval_pages(data, seed):
    rng = random.Random(seed + 200)
    likes, friends = like_sets(data), friend_sets(data)
    n_pages = len({p["id"] for p in data["pages"]})
    hidden = []
    for u in sorted(likes):
        if len(likes[u]) >= 3:
            p = rng.choice(sorted(likes[u]))
            likes[u].discard(p)
            hidden.append((u, p))
    res = {}
    for m in PAGE_METHODS:
        res[m] = summarise([ranks_of(score_pages(u, likes, friends, m), p) for u, p in hidden])
    res["random"] = random_baseline([n_pages - len(likes[u]) for u, _ in hidden])
    return res, len(hidden)


def check_integrity(clean):
    ids = {u["id"] for u in clean["users"]}
    pids = [p["id"] for p in clean["pages"]]
    fr = {u["id"]: set(u["friends"]) for u in clean["users"]}
    problems = 0
    problems += sum(1 for u in clean["users"] if not u["name"].strip())
    problems += sum(1 for u in clean["users"] if u["id"] in fr[u["id"]])
    problems += sum(1 for u in clean["users"] for f in u["friends"] if f not in ids or u["id"] not in fr[f])
    problems += sum(1 for u in clean["users"] if len(u["friends"]) != len(set(u["friends"])))
    problems += sum(1 for u in clean["users"] for p in u["liked_pages"] if p not in set(pids))
    problems += len(pids) - len(set(pids))
    problems += sum(1 for u in clean["users"] if not u["friends"] and not u["liked_pages"])
    return problems


def time_all_users(data, n_users):
    friends, likes = friend_sets(data), like_sets(data)
    t = time.perf_counter()
    for u in friends:
        score_people(u, friends, "adamic_adar")
        score_pages(u, likes, friends, "hybrid")
    return (time.perf_counter() - t) / n_users * 1000


def mean_std(dicts):
    keys = dicts[0].keys()
    return {k: (statistics.mean(d[k] for d in dicts), statistics.pstdev(d[k] for d in dicts)) for k in keys}


def main():
    os.makedirs(OUT, exist_ok=True)
    people_runs, page_runs, cleaning, n_hidden = [], [], [], {}
    for seed in SEEDS:
        data, _ = generate(N_USERS, seed=seed)
        p, n_hidden["people"] = eval_people(data, seed)
        g, n_hidden["pages"] = eval_pages(data, seed)
        people_runs.append(p)
        page_runs.append(g)
        dirty, _ = generate(N_USERS, dirty_fraction=0.10, seed=seed)
        clean, report = clean_data(dirty)
        cleaning.append({"issues_fixed": sum(v for k, v in report.items()), "report": report,
                         "integrity_violations_after": check_integrity(clean),
                         "users_in": len(dirty["users"]), "users_out": len(clean["users"])})

    people = {m: mean_std([r[m] for r in people_runs]) for m in people_runs[0]}
    pages = {m: mean_std([r[m] for r in page_runs]) for m in page_runs[0]}

    timing = {}
    for n in (1000, 5000, 10000):
        data, _ = generate(n, seed=0)
        timing[n] = time_all_users(data, n)

    data, _ = generate(N_USERS, seed=0)
    stats = {"users": len(data["users"]), "pages": len(data["pages"]),
             "friendships": sum(len(u["friends"]) for u in data["users"]) // 2,
             "likes": sum(len(u["liked_pages"]) for u in data["users"])}

    out = {"network": stats, "hidden_per_seed": n_hidden, "people_you_may_know": people,
           "pages_you_might_like": pages, "cleaning": cleaning, "ms_per_user": timing}
    with open(os.path.join(OUT, "evaluation.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    def table(res, best):
        rows = ["| Method | Hit@1 | Hit@5 | Hit@10 | MRR |", "|---|---|---|---|---|"]
        for m, r in res.items():
            cells = [f"{r[k][0]:.3f} ± {r[k][1]:.3f}" for k in ("hit@1", "hit@5", "hit@10", "mrr")]
            name = f"**{m}**" if m == best else m
            rows.append(f"| {name} | " + " | ".join(cells) + " |")
        return "\n".join(rows)

    best_p = max((m for m in people if m != "random"), key=lambda m: people[m]["hit@10"][0])
    best_g = max((m for m in pages if m != "random"), key=lambda m: pages[m]["hit@5"][0])
    md = [f"# Evaluation ({len(SEEDS)} seeds, {N_USERS} users, {stats['friendships']} friendships, "
          f"{stats['pages']} pages)", "",
          f"## People You May Know ({n_hidden['people']} hidden friendships per seed)", table(people, best_p), "",
          f"## Pages You Might Like ({n_hidden['pages']} hidden likes per seed)", table(pages, best_g), "",
          "## Cleaning (10% of users corrupted)",
          "| Seed | Issues fixed | Users in -> out | Integrity violations after |", "|---|---|---|---|"]
    md += [f"| {s} | {c['issues_fixed']} | {c['users_in']} -> {c['users_out']} | {c['integrity_violations_after']} |"
           for s, c in zip(SEEDS, cleaning)]
    md += ["", "## Speed (pure Python, both recommenders, every user)", "| Users | ms per user |", "|---|---|"]
    md += [f"| {n:,} | {t:.2f} |" for n, t in timing.items()]
    with open(os.path.join(OUT, "evaluation.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
