from pathlib import Path

from ire_assn1.data.mind import read_mind_articles, read_mind_behaviors


def test_mind_articles_repair_escaped_separators(tmp_path: Path) -> None:
    path = tmp_path / "news.tsv"
    path.write_text(
        "N1\tcat\tsub\tTitle\tAbstract\\thttps://example.test/article\\t[]\t[]\n"
        "N2\tcat\tsub\tTitle 2\\tAbstract 2\thttps://example.test/article-2\t"
        '[{"Label": "Name"}]\t[]\n',
        encoding="utf-8",
    )

    articles = read_mind_articles(path)

    assert [(article.title, article.abstract) for article in articles] == [
        ("Title", "Abstract"),
        ("Title 2", "Abstract 2"),
    ]
    assert articles[1].entities == ("Name",)


def test_mind_behaviors_accept_unlabeled_competition_impressions(tmp_path: Path) -> None:
    path = tmp_path / "behaviors.tsv"
    path.write_text(
        "7\tU1\t11/16/2019 12:00:00 PM\tN1\tN2 N3\n",
        encoding="utf-8",
    )
    behavior = read_mind_behaviors(path)[0]
    assert behavior.candidate_ids == ("N2", "N3")
    assert behavior.labels == ()
    assert behavior.clicked_ids == ()
