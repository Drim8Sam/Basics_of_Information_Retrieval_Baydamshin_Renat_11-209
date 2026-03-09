from pathlib import Path

from vector_search import (
    load_doc_urls,
    load_document_vectors,
    get_stemmer,
    search,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

TFIDF_DIR = PROJECT_ROOT / "Task4" / "lemmas_terms_per_doc"
INDEX_FILE = PROJECT_ROOT / "WebCrawler" / "index.txt"

DEFAULT_TOP_K = 10

def main() -> None:
    print("  Задание 5: Векторный поиск (TF-IDF + косинусная близость)")

    if not TFIDF_DIR.exists():
        raise SystemExit(
            f"Ошибка: директория {TFIDF_DIR} не найдена.\n"
        )

    print("\nЗагрузка данных...")

    print("[terms]  загрузка TF-IDF-векторов по терминам...")
    doc_vectors_terms, idf_terms = load_document_vectors(TFIDF_DIR, mode="terms")
    print(f"документов: {len(doc_vectors_terms)}, терминов в словаре: {len(idf_terms)}")

    print(" [lemmas] загрузка TF-IDF-векторов по леммам...")
    doc_vectors_lemmas, idf_lemmas = load_document_vectors(TFIDF_DIR, mode="lemmas")
    print(f"документов: {len(doc_vectors_lemmas)}, лемм в словаре: {len(idf_lemmas)}")

    stemmer = get_stemmer()

    doc_urls = load_doc_urls(INDEX_FILE)

    mode = "lemmas"
    top_k = DEFAULT_TOP_K

    print(f"\nРежим поиска: {mode}  |  Показывать top-{top_k} результатов")

    print("Инструкция:")
    print("Введите поисковый запрос на английском языке.")
    print("Команды:")
    print(":mode terms — переключить режим на термины")
    print(":mode lemmas — переключить режим на леммы")
    print(":top N — изменить количество результатов")
    print("exit / quit — выход")

    while True:
        try:
            raw = input("Запрос> ").strip()

            if not raw:
                continue

            if raw.lower() in ("exit", "quit", "q"):
                print("Выход из программы.")
                break

            if raw.lower().startswith(":mode"):
                parts = raw.split()
                if len(parts) == 2 and parts[1] in ("terms", "lemmas"):
                    mode = parts[1]
                    print(f"  Режим переключён на: {mode}\n")
                else:
                    print("  Использование: :mode terms  или  :mode lemmas\n")
                continue

            if raw.lower().startswith(":top"):
                parts = raw.split()
                if len(parts) == 2 and parts[1].isdigit() and int(parts[1]) > 0:
                    top_k = int(parts[1])
                    print(f"  Показывать top-{top_k} результатов\n")
                else:
                    print("  Использование: :top N  (N > 0)\n")
                continue

            if mode == "lemmas":
                results = search(
                    raw,
                    doc_vectors_lemmas,
                    idf_lemmas,
                    stemmer=stemmer,
                    top_k=top_k,
                )
            else:
                results = search(
                    raw,
                    doc_vectors_terms,
                    idf_terms,
                    stemmer=None,
                    top_k=top_k,
                )

            if not results:
                print("  Ничего не найдено.\n")
                continue

            print(f"\n  Найдено результатов (показаны top-{min(top_k, len(results))}"
                  f" из {len(results)}, режим={mode}):\n")

            print(f"  {'#':<4} {'Doc':>4}   {'Score':>8}   URL")
            print(f"  {'—'*4} {'—'*4}   {'—'*8}   {'—'*40}")

            for rank, (doc_id, score) in enumerate(results, start=1):
                url = doc_urls.get(doc_id, "—")
                print(f"  {rank:<4} {doc_id:>4}   {score:>8.5f}   {url}")

            print()

        except KeyboardInterrupt:
            print("\n\nВыход из программы.")
            break
        except Exception as e:
            print(f"\n  Ошибка: {e}\n")


if __name__ == "__main__":
    main()

