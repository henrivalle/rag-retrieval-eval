# RAG Retrieval Eval

[English](README.md) · [Como contribuir](CONTRIBUTING.md)

Compare recuperadores de documentos para RAG usando os mesmos julgamentos de relevância. Use Python 3.11 ou posterior. O exemplo em português usa documentos sintéticos e não precisa de dependências externas nem de acesso à rede.

```sh
python3 -m rag_retrieval_eval rank examples/corpus.json examples/queries.json --k 3 > /tmp/rag-rankings.json
python3 -m rag_retrieval_eval score examples/judgments.json /tmp/rag-rankings.json --k 2 \
  --baseline examples/reference-ranking.json
```

Os arquivos são objetos JSON em UTF-8. O corpus associa ID de documento ao texto. As consultas associam ID de consulta ao texto. Os julgamentos e rankings associam os mesmos IDs de consulta a listas de IDs de documentos sem duplicatas. Cada julgamento precisa de pelo menos um documento relevante. Uma lista vazia de resultados indica que o recuperador não retornou nada.

O relatório traz recall, posição recíproca média (MRR) e proporção de consultas com ao menos um acerto entre os `k` primeiros resultados. Os deltas usam candidato menos baseline. O `--min-recall` aceita frações de `0` a `1` e retorna código `1` se o limite falhar. Entradas inválidas retornam `2`.

## Comparação de rankings em dois cortes

A consulta sintética `q-aulas` tem dois documentos relevantes. O ranking salvo em `reference-ranking-comparison.json` representa outro recuperador: ele coloca `calendario` primeiro e deixa `aula-grafos` abaixo do corte `k=2`.

```sh
python3 -m rag_retrieval_eval rank examples/corpus.json examples/queries-comparison.json --k 3 > /tmp/rag-bm25-comparison.json
python3 -m rag_retrieval_eval score examples/judgments-comparison.json /tmp/rag-bm25-comparison.json --k 2 \
  --baseline examples/reference-ranking-comparison.json
python3 -m rag_retrieval_eval score examples/judgments-comparison.json /tmp/rag-bm25-comparison.json --k 3 \
  --baseline examples/reference-ranking-comparison.json
```

Resultados esperados para a consulta:

| Corte | BM25 Recall@k | BM25 MRR@k | Outro Recall@k | Outro MRR@k |
| --- | ---: | ---: | ---: | ---: |
| `k=2` | `1.0` | `1.0` | `0.5` | `0.5` |
| `k=3` | `1.0` | `1.0` | `1.0` | `0.5` |

O outro ranking recupera o segundo documento relevante no `k=3`, aumentando o recall, mas o primeiro relevante continua na segunda posição, então o MRR permanece `0.5`.
