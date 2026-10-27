import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from codebook import clean_data, pages_you_might_like, people_you_may_know  # noqa: E402
from codebook.recommend import score_people  # noqa: E402
from codebook.synthetic import generate  # noqa: E402

TOY = {
    "users": [
        {"id": 1, "name": "A", "friends": [2, 3], "liked_pages": [101]},
        {"id": 2, "name": "B", "friends": [1, 4], "liked_pages": [101, 102]},
        {"id": 3, "name": "C", "friends": [1, 4], "liked_pages": [103]},
        {"id": 4, "name": "D", "friends": [2, 3], "liked_pages": [102, 103]},
        {"id": 5, "name": "E", "friends": [4], "liked_pages": [104]},
    ],
    "pages": [{"id": i, "name": f"P{i}"} for i in (101, 102, 103, 104)],
}


class TestCleaning(unittest.TestCase):
    def test_rules(self):
        raw = {"users": [
            {"id": 1, "name": "A", "friends": [2, 2, 1, 99], "liked_pages": [101, 555]},
            {"id": 2, "name": "B", "friends": [], "liked_pages": []},
            {"id": 3, "name": " ", "friends": [1], "liked_pages": [101]},
            {"id": 4, "name": "D", "friends": [], "liked_pages": []},
            {"id": 1, "name": "A again", "friends": [], "liked_pages": []},
        ], "pages": [{"id": 101, "name": "X"}, {"id": 101, "name": "X dup"}]}
        clean, report = clean_data(raw)
        by_id = {u["id"]: u for u in clean["users"]}
        self.assertEqual(set(by_id), {1, 2})            # 3 blank name, 4 inactive, dup 1 dropped
        self.assertEqual(by_id[1]["friends"], [2])       # dedup, no self, no unknown
        self.assertEqual(by_id[2]["friends"], [1])       # symmetrised
        self.assertEqual(by_id[1]["liked_pages"], [101])
        self.assertEqual(len(clean["pages"]), 1)
        self.assertEqual(report["inactive_users_removed"], 1)
        self.assertEqual(raw["users"][0]["friends"], [2, 2, 1, 99])  # input untouched

    def test_synthetic_dirty_is_fully_cleaned(self):
        dirty, _ = generate(300, dirty_fraction=0.2, seed=3)
        clean, _ = clean_data(dirty)
        ids = {u["id"] for u in clean["users"]}
        fr = {u["id"]: set(u["friends"]) for u in clean["users"]}
        for u in clean["users"]:
            self.assertTrue(u["name"].strip())
            self.assertNotIn(u["id"], fr[u["id"]])
            for f in u["friends"]:
                self.assertIn(f, ids)
                self.assertIn(u["id"], fr[f])


class TestRecommendations(unittest.TestCase):
    def test_people_ranked_by_mutual_friends(self):
        # A and D share two mutual friends (B, C); E is not reachable through a friend of A
        self.assertEqual(people_you_may_know(1, TOY, "mutual"), [4])
        self.assertEqual(people_you_may_know(5, TOY, "mutual"), [2, 3])

    def test_never_suggests_self_or_existing_friend(self):
        data, _ = generate(200, seed=1)
        for m in ("mutual", "jaccard", "adamic_adar", "popularity"):
            for u in data["users"][:50]:
                recs = people_you_may_know(u["id"], data, m, k=20)
                self.assertNotIn(u["id"], recs)
                self.assertFalse(set(recs) & set(u["friends"]))

    def test_jaccard_bounded(self):
        friends = {u["id"]: set(u["friends"]) for u in TOY["users"]}
        self.assertTrue(all(0 < s <= 1 for s in score_people(1, friends, "jaccard").values()))

    def test_pages(self):
        # A likes 101; B also likes 101 and likes 102 -> suggest 102
        self.assertEqual(pages_you_might_like(1, TOY, "overlap"), [102])
        for m in ("overlap", "jaccard", "friends", "hybrid", "popularity"):
            self.assertNotIn(101, pages_you_might_like(1, TOY, m, k=10))

    def test_unknown_user(self):
        self.assertEqual(people_you_may_know(999, TOY), [])
        self.assertEqual(pages_you_might_like(999, TOY), [])


if __name__ == "__main__":
    unittest.main()
