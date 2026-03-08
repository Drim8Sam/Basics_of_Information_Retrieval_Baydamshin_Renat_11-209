from __future__ import annotations

import math
import sys
from collections import Counter, defaultdict
from pathlib import Path


# -------------------- Пути к данным и результатам --------------------
# PROJECT_ROOT — корень репозитория (директория уровнем выше папки Task4)
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# DATA_PATH — директория с выкачанными HTML-страницами (каждая страница сохранена в .txt)
DATA_PATH = PROJECT_ROOT / "WebCrawler" / "crawled_pages"

# TASK2_DIR — директория второго задания (используется для чтения результатов токенизации/лемматизации)
TASK2_DIR = PROJECT_ROOT / "Task2"
TOKENS_PATH = TASK2_DIR / "tokens.txt"
LEMMAS_PATH = TASK2_DIR / "lemmas.txt"
PER_DOC_TOKENS_DIR = TASK2_DIR / "tokens_per_doc"
PER_DOC_LEMMAS_DIR = TASK2_DIR / "lemmas_per_doc"

# OUTPUT_DIR — директория, куда сохраняются результаты TF-IDF для терминов и лемм по каждому документу
OUTPUT_DIR = PROJECT_ROOT / "Task4" / "lemmas_terms_per_doc"


def import_task2():
    """
    Подключает модуль Task2 к текущему скрипту.
    Зачем нужно:
    - переиспользовать одинаковые правила очистки текста и токенизации;
    - использовать общий список стоп-слов и фильтрацию “мусора”;
    - использовать функцию извлечения текста из HTML.
    Возвращает:
        модуль task2 (файл Task2/task2.py)
    """
    sys.path.insert(0, str(TASK2_DIR))
    import task2  # Task2/task2.py
    return task2


def idf_value(n_docs: int, df: int) -> float:
    """
    Вычисляет обратную документную частоту (IDF) для термина/леммы.
    Параметры:
        n_docs: общее число документов в коллекции
        df: число документов, в которых встречается термин/лемма
    Возвращает:
        значение IDF по формуле ln(N / (1 + df))
    """
    if n_docs <= 0:
        return 0.0
    return math.log(n_docs / (1.0 + df))


def write_lines(path: Path, lines: list[str]) -> None:
    """
    Сохраняет список строк в текстовый файл (UTF-8).
    Поведение:
    - создаёт директории по пути, если их нет;
    - записывает строки через перевод строки;
    - добавляет завершающий перевод строки, чтобы файл корректно воспринимался текстовыми утилитами.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def load_term_to_lemma(lemmas_txt: Path) -> dict[str, str]:
    """
    Загружает соответствие «токен -> лемма» из файла лемм.
    Ожидаемый формат файла:
        <лемма> <токен1> <токен2> ... <токенN>
    Возвращает:
        словарь, где ключ — токен, значение — лемма (основа/нормализованная форма)
    """
    mapping: dict[str, str] = {}
    if not lemmas_txt.exists():
        return mapping

    for line in lemmas_txt.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        lemma = parts[0].strip().lower()
        for tok in parts[1:]:
            t = tok.strip().lower()
            if t:
                mapping[t] = lemma

    return mapping


def compute_df_terms_from_task2(tokens_per_doc_dir: Path) -> dict[str, int]:
    """
    Считает документную частоту (DF) для терминов по результатам токенизации по документам.
    Вход:
        директория с файлами вида {id}_tokens.txt, где каждый файл содержит уникальные токены документа
        (по одному токену в строке).
    Логика:
        для каждого документа берётся множество токенов (уникальные),
        затем для каждого токена увеличивается счётчик DF на 1.
    Возвращает:
        словарь term -> df(term)
    """
    df: dict[str, int] = defaultdict(int)
    if not tokens_per_doc_dir.exists():
        return {}

    for f in sorted(tokens_per_doc_dir.glob("*_tokens.txt")):
        terms_in_doc = set(
            t.strip().lower()
            for t in f.read_text(encoding="utf-8", errors="ignore").splitlines()
            if t.strip()
        )
        for term in terms_in_doc:
            df[term] += 1

    return dict(df)


def compute_df_lemmas_from_task2(lemmas_per_doc_dir: Path) -> dict[str, int]:
    """
    Считает документную частоту (DF) для лемм по результатам группировки лемм по документам.
    Вход:
        директория с файлами вида {id}_lemmas.txt, где каждая строка начинается с леммы:
        <лемма> <токен1> <токен2> ...
    Логика:
        в каждом документе собирается множество лемм, затем для каждой леммы увеличивается DF на 1.
    Возвращает:
        словарь lemma -> df(lemma)
    """
    df: dict[str, int] = defaultdict(int)
    if not lemmas_per_doc_dir.exists():
        return {}

    for f in sorted(lemmas_per_doc_dir.glob("*_lemmas.txt")):
        lemmas_in_doc = set()
        for line in f.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = line.strip()
            if not line:
                continue
            lemma = line.split()[0].strip().lower()
            if lemma:
                lemmas_in_doc.add(lemma)

        for lemma in lemmas_in_doc:
            df[lemma] += 1

    return dict(df)


def tokenize_for_tf(text: str, task2_module, min_token_length: int = 3) -> list[str]:
    """
    Преобразует текст документа в список токенов (с сохранением повторов) для подсчёта TF.
    Отличие от “уникальных токенов”:
        здесь возвращается именно список, потому что TF требует количества вхождений.
    Правила отбора токенов:
    - выделяются последовательности латинских букв (ASCII);
    - приводятся к нижнему регистру;
    - отбрасываются стоп-слова;
    - отбрасываются “мусорные” токены;
    - минимальная длина токена задаётся параметром min_token_length (по умолчанию 3).
    Параметры:
        text: исходный текст документа
        task2_module: модуль с общими настройками (regex, стоп-слова, фильтры)
        min_token_length: минимальная длина токена
    Возвращает:
        список токенов (с повторениями)
    """
    out: list[str] = []
    for m in task2_module.TOKEN_RE.finditer(text):
        tok = m.group(0).lower()

        if len(tok) < min_token_length:
            continue
        if tok in task2_module.STOP_WORDS:
            continue
        if not tok.isalpha():
            continue
        if task2_module.is_noise_token(tok):
            continue

        out.append(tok)

    return out


def main() -> None:
    """
    Запускает полный расчёт TF-IDF для терминов и лемм по каждому документу.
    Шаги:
    1) Проверяет наличие входных данных и результатов предыдущей обработки.
    2) Загружает модуль с общей логикой извлечения текста/фильтрации.
    3) Считает DF для терминов и DF для лемм по данным предыдущего задания.
    4) Для каждого документа:
       - извлекает текст из HTML;
       - строит список токенов (для TF);
       - считает TF-IDF по терминам и записывает файл {id}_terms.txt;
       - агрегирует TF по леммам как сумму частот терминов одной леммы;
       - считает TF-IDF по леммам и записывает файл {id}_lemmas.txt.
    """
    if not DATA_PATH.exists():
        raise FileNotFoundError("Не найдена директория WebCrawler/crawled_pages")

    if not PER_DOC_TOKENS_DIR.exists() or not PER_DOC_LEMMAS_DIR.exists():
        raise FileNotFoundError(
            "Не найдены результаты Task2 (tokens_per_doc / lemmas_per_doc). Сначала запусти Task2."
        )

    task2 = import_task2()
    stemmer = task2.get_porter_stemmer()

    files = sorted(DATA_PATH.glob("*.txt"), key=lambda p: p.name)
    n_docs = len(files)

    # DF и IDF считаются на основе того, в скольких документах встретился термин/лемма.
    df_terms = compute_df_terms_from_task2(PER_DOC_TOKENS_DIR)
    df_lemmas = compute_df_lemmas_from_task2(PER_DOC_LEMMAS_DIR)

    # Соответствие «термин -> лемма» используется для корректной агрегации TF по леммам.
    term_to_lemma = load_term_to_lemma(LEMMAS_PATH)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for file in files:
        doc_id = file.stem

        text = task2.extract_text_from_html_file(file)
        terms_list = tokenize_for_tf(text, task2, min_token_length=3)

        total_terms = len(terms_list)
        terms_out_path = OUTPUT_DIR / f"{doc_id}_terms.txt"
        lemmas_out_path = OUTPUT_DIR / f"{doc_id}_lemmas.txt"

        if total_terms == 0:
            write_lines(terms_out_path, [])
            write_lines(lemmas_out_path, [])
            continue

        term_counts = Counter(terms_list)

        # Термины: рассчитываются TF и TF-IDF для каждого термина.
        term_lines: list[str] = []
        for term in sorted(term_counts.keys()):
            tf = term_counts[term] / total_terms
            idf = idf_value(n_docs, df_terms.get(term, 0))
            tfidf = tf * idf
            term_lines.append(f"{term} {idf:.6f} {tfidf:.6f}")
        write_lines(terms_out_path, term_lines)

        # Леммы: TF(леммы) = сумма частот терминов, относящихся к этой лемме, делённая на число терминов в документе.
        lemma_counts: dict[str, int] = defaultdict(int)

        for term, cnt in term_counts.items():
            lemma = term_to_lemma.get(term)
            if not lemma:
                lemma = stemmer.stem(term)
            lemma = lemma.strip().lower()

            # Фильтрация некорректных лемм
            if not lemma:
                continue
            if not lemma.isascii():
                continue
            if len(lemma) >= 5 and len(set(lemma)) == 1:
                continue
            if len(lemma) > 30:
                continue
            if lemma in task2.STOP_WORDS:
                continue

            lemma_counts[lemma] += cnt

        lemma_lines: list[str] = []
        for lemma in sorted(lemma_counts.keys()):
            tf = lemma_counts[lemma] / total_terms
            idf = idf_value(n_docs, df_lemmas.get(lemma, 0))
            tfidf = tf * idf
            lemma_lines.append(f"{lemma} {idf:.6f} {tfidf:.6f}")
        write_lines(lemmas_out_path, lemma_lines)

    print(f"Готово. Документов: {n_docs}")
    print(f"Выход: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()