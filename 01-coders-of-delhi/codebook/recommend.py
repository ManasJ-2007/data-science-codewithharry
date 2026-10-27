"""'People You May Know' and 'Pages You Might Like'.

People (link prediction) scores for a non-friend candidate v of user u, with N(x) = friends of x:
    mutual        |N(u) & N(v)|                            (count of mutual friends)
    jaccard       |N(u) & N(v)| / |N(u) | N(v)|            (mutual friends, normalised by network size)
    adamic_adar   sum over z in N(u) & N(v) of 1/log|N(z)| (a mutual friend with few friends counts more)
    popularity    |N(v)|                                   (baseline: most-connected users)

Pages (collaborative filtering) scores for a page p not yet liked by u, with L(x) = pages liked by x:
    overlap       sum over users w who like p of |L(u) & L(w)|
    jaccard       sum over users w who like p of |L(u) & L(w)| / |L(u) | L(w)|
    friends       number of u's friends who like p
    hybrid        jaccard + 0.5 * friends / |N(u)|
    popularity    number of users who like p                (baseline)
Ties are broken by id so results are deterministic.
"""
import math
from collections import defaultdict

from .data import friend_sets, like_sets

PEOPLE_METHODS = ("mutual", "jaccard", "adamic_adar", "popularity")
PAGE_METHODS = ("overlap", "jaccard", "friends", "hybrid", "popularity")


def _rank(scores, k):
    ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    return [item for item, s in ranked if s > 0][:k] if k else [item for item, s in ranked if s > 0]


def score_people(user_id, friends, method="mutual"):
    """friends: user id -> set of friend ids. Returns {candidate: score}."""
    if user_id not in friends:
        return {}
    mine = friends[user_id]
    if method == "popularity":
        return {v: len(fs) for v, fs in friends.items() if v != user_id and v not in mine}
    scores = defaultdict(float)
    for z in mine:
        weight = 1.0 / math.log(len(friends[z])) if method == "adamic_adar" and len(friends[z]) > 1 else 1.0
        for v in friends[z]:
            if v != user_id and v not in mine:
                scores[v] += weight
    if method == "jaccard":
        for v in scores:
            scores[v] /= len(mine | friends[v])
    elif method not in ("mutual", "adamic_adar"):
        raise ValueError(f"unknown method {method!r}; choose from {PEOPLE_METHODS}")
    return dict(scores)


def score_pages(user_id, likes, friends=None, method="overlap"):
    """likes: user id -> set of liked page ids. Returns {page: score}."""
    if user_id not in likes:
        return {}
    mine = likes[user_id]
    scores = defaultdict(float)
    if method == "popularity":
        for w, pages in likes.items():
            if w != user_id:
                for p in pages - mine:
                    scores[p] += 1
        return dict(scores)
    if method in ("overlap", "jaccard", "hybrid"):
        for w, pages in likes.items():
            if w == user_id:
                continue
            shared = len(mine & pages)
            if not shared:
                continue
            sim = shared if method == "overlap" else shared / len(mine | pages)
            for p in pages - mine:
                scores[p] += sim
    if method in ("friends", "hybrid"):
        my_friends = (friends or {}).get(user_id, set())
        boost = 1.0 if method == "friends" else 0.5 / max(len(my_friends), 1)
        for f in my_friends:
            for p in likes.get(f, set()) - mine:
                scores[p] += boost
    if method not in PAGE_METHODS:
        raise ValueError(f"unknown method {method!r}; choose from {PAGE_METHODS}")
    return dict(scores)


def people_you_may_know(user_id, data, method="mutual", k=5):
    """Top-k suggested user ids for `user_id`."""
    return _rank(score_people(user_id, friend_sets(data), method), k)


def pages_you_might_like(user_id, data, method="overlap", k=5):
    """Top-k suggested page ids for `user_id`."""
    return _rank(score_pages(user_id, like_sets(data), friend_sets(data), method), k)
