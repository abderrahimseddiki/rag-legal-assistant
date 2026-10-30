# Données

Les documents ne sont pas versionnés (voir `.gitignore`). Place-les dans `data/raw/`.

## Sources publiques suggérées

| Corpus | Où le trouver | Format |
|---|---|---|
| Code civil belge (Livres 1, 3, 5, 6, 8) | Justel – SPF Justice (ejustice.just.fgov.be), section « Législation consolidée » | HTML / PDF |
| Code pénal belge | Justel | HTML / PDF |
| Règlement des études HE2b | Site de la HE2b (règlement général des études) | PDF |

Formats acceptés : `.pdf`, `.html`, `.htm`, `.txt`, `.md`.

## Jeu d'évaluation

Copie `eval/questions.example.jsonl` vers `eval/questions.jsonl` et écris tes propres questions
(objectif : ~200) à partir des documents. Pour chaque question :

- `expected_source` : nom du fichier qui contient la réponse ;
- `expected_article` : numéro d'article attendu (ou `null`) ;
- `reference_answer` : la réponse correcte, rédigée par toi ;
- `answerable: false` pour ~15 % de questions volontairement hors corpus
  (elles mesurent la capacité du système à refuser au lieu d'inventer).

Écrire ces questions toi-même est essentiel : ce sont elles qui rendent tes chiffres crédibles.
