"""WEB DEMO для векторного поиска по документам (TF-IDF + cosine similarity)."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from flask import Flask, render_template_string, request

# Базовые пути проекта (скрипт лежит в DEMO/, поэтому берем корень через parents[1]).
PROJECT_ROOT = Path(__file__).resolve().parents[1]
TASK5_DIR = PROJECT_ROOT / "Task5"
TFIDF_DIR = PROJECT_ROOT / "Task4" / "lemmas_terms_per_doc"
INDEX_FILE = PROJECT_ROOT / "WebCrawler" / "index.txt"
DEFAULT_TOP_K = 10

# Добавляем Task5 в sys.path, чтобы переиспользовать готовый движок поиска.
if str(TASK5_DIR) not in sys.path:
    sys.path.insert(0, str(TASK5_DIR))

from vector_search import get_stemmer, load_doc_urls, load_document_vectors, search

app = Flask(__name__)


def load_search_data() -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Загружает векторы и URL-индекс при старте приложения.

    Возвращает:
    - data: словарь с данными поиска при успешной загрузке
    - error: текст ошибки (или нескольких ошибок), если что-то не найдено
    """
    errors: List[str] = []

    if not TFIDF_DIR.exists() or not TFIDF_DIR.is_dir():
        errors.append(f"Не найдена папка TF-IDF: {TFIDF_DIR}")

    terms_vectors: Dict[int, Dict[str, float]] = {}
    terms_idf: Dict[str, float] = {}
    lemmas_vectors: Dict[int, Dict[str, float]] = {}
    lemmas_idf: Dict[str, float] = {}

    if not errors:
        # Загружаем два индекса: terms и lemmas.
        terms_vectors, terms_idf = load_document_vectors(TFIDF_DIR, mode="terms")
        lemmas_vectors, lemmas_idf = load_document_vectors(TFIDF_DIR, mode="lemmas")

        if not terms_vectors:
            errors.append(
                f"Не найдены файлы TF-IDF для terms в {TFIDF_DIR} (ожидаются *_terms.txt)."
            )
        if not lemmas_vectors:
            errors.append(
                f"Не найдены файлы TF-IDF для lemmas в {TFIDF_DIR} (ожидаются *_lemmas.txt)."
            )

    if not INDEX_FILE.exists() or not INDEX_FILE.is_file():
        errors.append(f"Не найден файл индекса URL: {INDEX_FILE}")
        doc_urls: Dict[int, str] = {}
    else:
        # Сопоставление doc_id -> URL из WebCrawler/index.txt.
        doc_urls = load_doc_urls(INDEX_FILE)

    if errors:
        return None, "\n".join(errors)

    data = {
        "terms_vectors": terms_vectors,
        "terms_idf": terms_idf,
        "lemmas_vectors": lemmas_vectors,
        "lemmas_idf": lemmas_idf,
        "doc_urls": doc_urls,
        "stemmer": get_stemmer(),
    }
    return data, None


# Загружаем индекс один раз при старте, чтобы не читать файлы на каждый запрос.
SEARCH_DATA, STARTUP_ERROR = load_search_data()

# Встроенный HTML-шаблон страницы поиска.
HTML_TEMPLATE = """
<!doctype html>
<html lang="ru">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Векторный поиск по документам</title>
  <style>
    :root {
      --bg: #f4f7fb;
      --card: #ffffff;
      --text: #18212f;
      --muted: #5a6779;
      --accent: #0f766e;
      --accent-hover: #0b5f58;
      --border: #d7dde7;
      --error-bg: #fff1f1;
      --error-text: #8b1e1e;
    }

    * { box-sizing: border-box; }

    body {
      margin: 0;
      font-family: "Segoe UI", Tahoma, sans-serif;
      color: var(--text);
      background: radial-gradient(circle at top right, #eaf4ff, var(--bg));
    }

    .container {
      max-width: 980px;
      margin: 32px auto;
      padding: 24px;
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 14px;
      box-shadow: 0 10px 24px rgba(24, 33, 47, 0.06);
    }

    h1 {
      margin: 0 0 6px 0;
      font-size: 30px;
    }

    .subtitle {
      margin: 0 0 20px 0;
      color: var(--muted);
      font-size: 15px;
    }

    .error {
      padding: 12px 14px;
      margin-bottom: 18px;
      border: 1px solid #ffc9c9;
      border-radius: 10px;
      background: var(--error-bg);
      color: var(--error-text);
      white-space: pre-line;
    }

    form {
      display: grid;
      grid-template-columns: 1fr 180px auto;
      gap: 10px;
      margin-bottom: 14px;
    }

    input[type="text"] {
      width: 100%;
      padding: 11px 12px;
      border: 1px solid var(--border);
      border-radius: 10px;
      font-size: 16px;
    }

    select {
      width: 100%;
      padding: 11px 12px;
      border: 1px solid var(--border);
      border-radius: 10px;
      font-size: 15px;
      background: #fff;
    }

    button {
      padding: 11px 18px;
      border: 0;
      border-radius: 10px;
      font-size: 15px;
      color: #fff;
      background: var(--accent);
      cursor: pointer;
    }

    button:hover {
      background: var(--accent-hover);
    }

    .meta {
      color: var(--muted);
      font-size: 14px;
      margin: 10px 0;
    }

    table {
      width: 100%;
      border-collapse: collapse;
      margin-top: 6px;
      overflow: hidden;
      border-radius: 10px;
      border: 1px solid var(--border);
    }

    th, td {
      padding: 10px 12px;
      border-bottom: 1px solid var(--border);
      text-align: left;
      vertical-align: top;
    }

    th {
      background: #f7fafc;
      font-weight: 600;
    }

    tr:last-child td {
      border-bottom: 0;
    }

    .score {
      font-variant-numeric: tabular-nums;
    }

    a {
      color: #0f4ca3;
      text-decoration: none;
      word-break: break-word;
    }

    a:hover {
      text-decoration: underline;
    }

    @media (max-width: 700px) {
      .container {
        margin: 12px;
        padding: 16px;
      }

      form {
        grid-template-columns: 1fr;
      }
    }
  </style>
</head>
<body>
  <main class="container">
    <h1>Векторный поиск по документам</h1>
    <p class="subtitle">TF-IDF + косинусная близость</p>

    {% if startup_error %}
      <div class="error">{{ startup_error }}</div>
    {% endif %}

    <form method="get" action="/">
      <input
        type="text"
        name="q"
        value="{{ query|e }}"
        placeholder="Введите поисковый запрос"
        {% if startup_error %}disabled{% endif %}
      />
      <select name="mode" {% if startup_error %}disabled{% endif %}>
        <option value="lemmas" {% if mode == "lemmas" %}selected{% endif %}>По леммам</option>
        <option value="terms" {% if mode == "terms" %}selected{% endif %}>По терминам</option>
      </select>
      <button type="submit" {% if startup_error %}disabled{% endif %}>Search</button>
    </form>

    {% if show_results and not startup_error %}
      <p class="meta">Показаны top-{{ top_k }} результатов. Найдено: {{ found_count }}.</p>

      {% if results %}
        <table>
          <thead>
            <tr>
              <th>#</th>
              <th>doc_id</th>
              <th>score</th>
              <th>url</th>
            </tr>
          </thead>
          <tbody>
            {% for item in results %}
              <tr>
                <td>{{ item.rank }}</td>
                <td>{{ item.doc_id }}</td>
                <td class="score">{{ "%.6f"|format(item.score) }}</td>
                <td>
                  {% if item.url %}
                    <a href="{{ item.url|e }}" target="_blank" rel="noopener noreferrer">{{ item.url }}</a>
                  {% else %}
                    URL не найден
                  {% endif %}
                </td>
              </tr>
            {% endfor %}
          </tbody>
        </table>
      {% else %}
        <p>Ничего не найдено</p>
      {% endif %}
    {% endif %}
  </main>
</body>
</html>
"""


def build_results(query: str, mode: str, top_k: int) -> List[Dict[str, Any]]:
    """Выполняет поиск через Task5.search и готовит данные для таблицы."""
    if not SEARCH_DATA:
        return []

    if mode == "terms":
        # Режим terms: сравниваем исходные термины без стемминга.
        raw_results = search(
            query=query,
            doc_vectors=SEARCH_DATA["terms_vectors"],
            global_idf=SEARCH_DATA["terms_idf"],
            stemmer=None,
            top_k=top_k,
        )
    else:
        # Режим lemmas: применяем стеммер, как в Task5.
        raw_results = search(
            query=query,
            doc_vectors=SEARCH_DATA["lemmas_vectors"],
            global_idf=SEARCH_DATA["lemmas_idf"],
            stemmer=SEARCH_DATA["stemmer"],
            top_k=top_k,
        )

    rows: List[Dict[str, Any]] = []
    for rank, (doc_id, score) in enumerate(raw_results, start=1):
        rows.append(
            {
                "rank": rank,
                "doc_id": doc_id,
                "score": score,
                "url": SEARCH_DATA["doc_urls"].get(doc_id, ""),
            }
        )
    return rows


@app.get("/")
def home() -> str:
    """Контроллер: читает query, запускает поиск и рендерит страницу."""
    # Параметры из URL: q (текст запроса) и mode (lemmas/terms).
    query = (request.args.get("q") or "").strip()
    mode = (request.args.get("mode") or "lemmas").strip().lower()
    if mode not in {"lemmas", "terms"}:
        mode = "lemmas"

    results: List[Dict[str, Any]] = []
    show_results = bool(query)

    # Пустой запрос не ломает страницу: показываем только форму.
    if show_results and not STARTUP_ERROR:
        results = build_results(query=query, mode=mode, top_k=DEFAULT_TOP_K)

    return render_template_string(
        HTML_TEMPLATE,
        startup_error=STARTUP_ERROR,
        query=query,
        mode=mode,
        top_k=DEFAULT_TOP_K,
        show_results=show_results,
        results=results,
        found_count=len(results),
    )


if __name__ == "__main__":
    # Локальный запуск: python DEMO/web_demo.py
    app.run(host="127.0.0.1", port=5000, debug=True)
