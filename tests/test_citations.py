from rag_assistant.citations import NOT_FOUND, Source, build_context, extract_citations, validate


def test_extract_citations_dedup_in_order():
    assert extract_citations("A [2]. B [1][2]. C [3].") == [2, 1, 3]


def test_validate_grounded_answer():
    r = validate("Le délai est de 15 jours [1].", n_sources=3)
    assert r["grounded"] and r["invalid_citations"] == []


def test_validate_detects_invented_source():
    r = validate("Selon [7], c'est interdit.", n_sources=3)
    assert not r["grounded"] and r["invalid_citations"] == [7]


def test_validate_uncited_answer_is_not_grounded():
    assert not validate("C'est autorisé.", n_sources=3)["grounded"]


def test_abstention_counts_as_grounded():
    r = validate(NOT_FOUND, n_sources=0)
    assert r["abstained"] and r["grounded"]


def test_build_context_labels():
    s = Source(1, "loi.pdf", 4, "12", "Texte de l'article.")
    ctx = build_context([s])
    assert ctx.startswith("[1] (loi.pdf, art. 12, p. 4)")
