from __future__ import annotations

import re
from pathlib import Path
from html.parser import HTMLParser
from html import unescape

# Набор английских стоп-слов и частых HTML-энтити.
# Используется для удаления служебных слов (союзы/предлоги и т.п.) и явного мусора из токенов.
STOP_WORDS = {
    "a","about","above","after","again","against","all","am","an","and","any","are","as","at",
    "be","because","been","before","being","below","between","both","but","by",
    "can","could",
    "did","do","does","doing","down","during",
    "each",
    "few","for","from","further",
    "had","has","have","having","he","her","here","hers","herself","him","himself","his","how",
    "i","if","in","into","is","it","its","itself",
    "just",
    "me","more","most","my","myself",
    "no","nor","not","now",
    "of","off","on","once","only","or","other","our","ours","ourselves","out","over","own",
    "same","she","should","so","some","such",
    "than","that","the","their","theirs","them","themselves","then","there","these","they",
    "this","those","through","to","too",
    "under","until","up",
    "very",
    "was","we","were","what","when","where","which","while","who","whom","why","will","with","would",
    "you","your","yours","yourself","yourselves",

    # частые HTML-энтити как “мусор”
    "nbsp","amp","lt","gt","quot","apos",
}

# Регулярное выражение для выделения “чистых” слов:
# - только буквенные последовательности (a-z),
# - без цифр и смешанных “abc123”.
TOKEN_RE = re.compile(r"[a-z]+", re.IGNORECASE | re.ASCII)


class _HtmlTextExtractor(HTMLParser):
    """
    Парсер HTML, извлекающий только текстовое содержимое.

    Особенности:
    - пропускает содержимое тегов script/style/noscript;
    - собирает текстовые фрагменты и возвращает их одной строкой.
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._chunks: list[str] = []
        self._skip = 0  # счётчик “глубины” внутри script/style/noscript

    def handle_starttag(self, tag, attrs):
        # Внутри этих тегов обычно находится нерелевантный текст (код/стили),
        # поэтому игнорируем всё, что будет встречено до закрывающего тега.
        if tag.lower() in {"script", "style", "noscript"}:
            self._skip += 1

    def handle_endtag(self, tag):
        # Завершаем режим пропуска, когда дошли до закрывающего тега.
        if tag.lower() in {"script", "style", "noscript"} and self._skip > 0:
            self._skip -= 1

    def handle_data(self, data):
        # В обычном режиме сохраняем текстовые фрагменты.
        if self._skip == 0 and data:
            self._chunks.append(data)

    def text(self) -> str:
        """Возвращает собранный текст одной строкой."""
        return " ".join(self._chunks)


def extract_text_from_html_file(path: Path) -> str:
    """
    Читает файл с HTML-разметкой и возвращает извлечённый текст.

    - Файл читается как UTF-8 с игнорированием некорректных символов.
    - HTML-энтити декодируются (например, &amp; -> &).
    """
    html = path.read_text(encoding="utf-8", errors="ignore")
    parser = _HtmlTextExtractor()
    parser.feed(html)
    return unescape(parser.text())

def is_noise_token(tok: str) -> bool:
    """
    Фильтр “мусора”:
    - не-ASCII токены (например, турецкие буквы);
    - повтор одного символа (aaaaa, bbbbbbb);
    - чрезмерно длинные обрывки.
    """
    if not tok.isascii():
        return True
    if len(tok) >= 5 and len(set(tok)) == 1:
        return True
    if len(tok) > 30:
        return True
    return False

def tokenize(text: str, min_token_length: int = 2) -> set[str]:
    """
    Токенизация текста:
    - выделяются ASCII-слова по regex [a-z]+;
    - приводятся к нижнему регистру;
    - отбрасываются короткие, стоп-слова и “мусорные” токены;
    - результат без дубликатов.
    """
    tokens: set[str] = set()

    for m in TOKEN_RE.finditer(text):
        tok = m.group(0).lower()

        if len(tok) < min_token_length:
            continue
        if tok in STOP_WORDS:
            continue
        if not tok.isalpha():  # дополнительная страховка
            continue
        if is_noise_token(tok):
            continue

        tokens.add(tok)

    return tokens


def get_porter_stemmer():
    """
    Создаёт стеммер Портера для нормализации слов к основе.

    Требование: установлен пакет nltk.
    """
    from nltk.stem import PorterStemmer
    return PorterStemmer()


def get_porter_stemmer():
    """Создаёт стеммер Портера (нужен пакет nltk)."""
    from nltk.stem import PorterStemmer
    return PorterStemmer()


def group_tokens_by_stem(tokens: set[str], stemmer) -> dict[str, list[str]]:
    """
    Группирует токены по основе (стему).
    Возвращает: stem -> [tokens...]
    """
    groups: dict[str, list[str]] = {}

    for t in tokens:
        lemma = stemmer.stem(t)

        if not lemma or lemma.strip() == "":
            continue
        # Стоп-слова среди ключей тоже отсекаем.
        if lemma.lower() in STOP_WORDS:
            continue
        # Защита от мусора в ключах.
        if not lemma.isascii():
            continue
        if len(lemma) >= 5 and len(set(lemma)) == 1:
            continue
        if len(lemma) > 30:
            continue

        groups.setdefault(lemma, []).append(t)

    return groups


def write_lines(path: Path, lines: list[str]) -> None:
    """
    Сохраняет строки в файл в кодировке UTF-8.
    - создаёт родительские директории при необходимости;
    - добавляет завершающий перевод строки (как это обычно ожидается в текстовых файлах).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def main() -> None:
    """
    Основной пайплайн обработки:
    1) Для каждого документа:
       - извлечь текст из HTML;
       - токенизировать (уникальные токены документа);
       - сохранить токены документа (по одному в строке);
       - сгруппировать токены по основам и сохранить группы.
    2) Параллельно накопить глобальные:
       - множество уникальных токенов корпуса;
       - словарь основ -> множество токенов.
    3) Сохранить глобальные файлы:
       - tokens.txt
       - lemmas.txt
    """
    project_root = Path(__file__).resolve().parents[1]
    data_path = project_root / "WebCrawler" / "crawled_pages"

    output_dir = project_root / "Task2"
    tokens_path = output_dir / "tokens.txt"
    lemmas_path = output_dir / "lemmas.txt"
    per_doc_tokens_dir = output_dir / "tokens_per_doc"
    per_doc_lemmas_dir = output_dir / "lemmas_per_doc"

    per_doc_tokens_dir.mkdir(parents=True, exist_ok=True)
    per_doc_lemmas_dir.mkdir(parents=True, exist_ok=True)

    stemmer = get_porter_stemmer()

    all_tokens: set[str] = set()
    all_lemmas: dict[str, set[str]] = {}

    files = sorted(data_path.glob("*.txt"), key=lambda p: p.name)

    for file in files:
        file_name = file.stem

        text = extract_text_from_html_file(file)
        doc_tokens = tokenize(text, min_token_length=2)

        if not doc_tokens:
            continue

        # Файл токенов текущего документа: по одному токену в строке (отсортировано).
        per_doc_tokens_file = per_doc_tokens_dir / f"{file_name}_tokens.txt"
        write_lines(per_doc_tokens_file, sorted(doc_tokens))

        # Группы токенов по основе для текущего документа.
        doc_lemmas = group_tokens_by_stem(doc_tokens, stemmer)

        # Формат строк: "<основа> <токен1> <токен2> ..."
        per_doc_lemma_lines = []
        for lemma in sorted(doc_lemmas.keys()):
            toks_sorted = sorted(set(doc_lemmas[lemma]))
            per_doc_lemma_lines.append(f"{lemma} " + " ".join(toks_sorted))

        per_doc_lemmas_file = per_doc_lemmas_dir / f"{file_name}_lemmas.txt"
        write_lines(per_doc_lemmas_file, per_doc_lemma_lines)

        # Накопление глобальных структур.
        all_tokens |= doc_tokens
        for lemma, toks in doc_lemmas.items():
            all_lemmas.setdefault(lemma, set()).update(toks)

    # Глобальный список токенов корпуса.
    write_lines(tokens_path, sorted(all_tokens))

    # Глобальные группы: "<основа> <токен1> <токен2> ..."
    lemma_lines = []
    for lemma in sorted(all_lemmas.keys()):
        toks_sorted = sorted(all_lemmas[lemma])
        lemma_lines.append(f"{lemma} " + " ".join(toks_sorted))
    write_lines(lemmas_path, lemma_lines)

    print(f"Готово! Токенов: {len(all_tokens)}, Групп (лемм): {len(all_lemmas)}")
    print(f"Вход:  {data_path}")
    print(f"Выход: {output_dir}")


if __name__ == "__main__":
    main()