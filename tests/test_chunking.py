from rag_assistant.chunking import Chunk, chunk_id, chunk_page, normalize, split_by_article, split_long

LAW = """TITRE I. Dispositions générales.

Art. 1er. La présente loi règle une matière visée à l'article 74 de la Constitution.

Art. 2. Pour l'application de la présente loi, on entend par étudiant toute personne inscrite.

Article 3/1. Le recours est introduit dans les quinze jours.
"""


def test_split_by_article_detects_articles_and_preamble():
    parts = split_by_article(normalize(LAW))
    assert parts[0][0] is None and "TITRE I" in parts[0][1]
    assert [p[0] for p in parts[1:]] == ["1er", "2", "3/1"]


def test_text_without_articles_is_one_part():
    assert split_by_article("Un simple paragraphe.") == [(None, "Un simple paragraphe.")]


def test_chunk_page_keeps_metadata():
    chunks = chunk_page(LAW, {"source": "loi.pdf", "page": 3})
    art2 = [c for c in chunks if c.metadata.get("article") == "2"][0]
    assert art2.metadata["source"] == "loi.pdf"
    assert art2.metadata["page"] == 3
    assert art2.text.startswith("Art. 2.")


def test_split_long_respects_max_and_never_cuts_words():
    sentence = "Le conseil statue sur le recours dans un délai raisonnable. "
    text = sentence * 80
    pieces = split_long(text, max_chars=300, overlap=50)
    assert len(pieces) > 1
    assert all(len(p) <= 300 for p in pieces)
    words = set(sentence.split())
    for p in pieces:
        assert set(p.split()) <= words


def test_normalize_rejoins_hyphenated_words():
    assert normalize("respon-\nsabilité") == "responsabilité"


def test_chunk_id_is_stable_and_distinct():
    a = Chunk("texte", {"source": "a.pdf", "page": 1, "article": "1", "part": 0})
    b = Chunk("texte", {"source": "a.pdf", "page": 2, "article": "1", "part": 0})
    assert chunk_id(a) == chunk_id(a)
    assert chunk_id(a) != chunk_id(b)
