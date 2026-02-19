import json
import time
import re
from pathlib import Path
from typing import List, Tuple, Optional

import requests
from requests import Response, Session


# ----------------------------
# НАСТРОЙКИ
# ----------------------------
URLS_FILE = "urls.json"             # входной список URL ({"urls":[...]})
OUTPUT_DIR = "crawled_pages"        # папка для выкачки
INDEX_PATH = "index.txt"            # индекс: номер -> URL

MAX_PAGES = 100                     # нужно минимум 100 страниц
REQUEST_DELAY_SEC = 0.2             # пауза между запросами
TIMEOUT_SEC = 15                    # таймаут на запрос
RETRIES = 3                         # повторные попытки при сетевых ошибках

# чтобы не тащить в список "ресурсы" (js/css/картинки/архивы и т.п.)
DISALLOWED_EXTENSIONS = {
    ".js", ".css", ".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp",
    ".ico", ".pdf", ".zip", ".rar", ".7z", ".mp3", ".mp4", ".avi", ".mov",
    ".woff", ".woff2", ".ttf", ".eot"
}


# ----------------------------
# УТИЛИТЫ
# ----------------------------
def is_bad_resource_url(url: str) -> bool:
    """
    Отсекаем ссылки, которые явно ведут на НЕ HTML-страницы (по расширению).
    По ТЗ список должен быть именно страницами, а не ресурсами.
    """
    u = url.lower().split("?")[0].split("#")[0]
    for ext in DISALLOWED_EXTENSIONS:
        if u.endswith(ext):
            return True
    return False


def normalize_urls(urls: List[str]) -> List[str]:
    """
    Убираем пустые, дубликаты, и очевидно неверные/неподходящие URL.
    Сохраняем порядок.
    """
    seen = set()
    out: List[str] = []
    for u in urls:
        if not isinstance(u, str):
            continue
        u = u.strip()
        if not u:
            continue
        if is_bad_resource_url(u):
            continue
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out


# ----------------------------
# IO: загрузка списка URL
# ----------------------------
def load_urls(file_path: str) -> List[str]:
    """
    Читает urls.json формата:
    {
      "urls": [
        "https://ru.wikipedia.org/wiki/Алгоритм",
        ...
      ]
    }
    """
    path = Path(file_path)
    data = json.loads(path.read_text(encoding="utf-8"))
    raw_urls = data.get("urls", [])
    if not isinstance(raw_urls, list):
        raise ValueError("urls.json: поле 'urls' должно быть массивом строк")
    return normalize_urls(raw_urls)


# ----------------------------
# HTTP: создание сессии и скачивание
# ----------------------------
def create_http_session() -> Session:
    """
    - единая сессия на всё выполнение
    - User-Agent под браузер
    """
    s = requests.Session()
    s.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "ru-RU,ru;q=0.9,en;q=0.8",
        "Connection": "keep-alive",
    })
    return s


def is_html_response(resp: Response) -> bool:
    """
    Проверка Content-Type. Нужны текстовые HTML-страницы.
    """
    ctype = (resp.headers.get("Content-Type") or "").lower()
    return "text/html" in ctype or "application/xhtml+xml" in ctype


def fetch_html(session: Session, url: str) -> Optional[str]:
    """
    Скачиваем HTML.
    По ТЗ: НЕ очищаем от разметки — сохраняем как есть.
    Возвращаем текст страницы или None при ошибке.
    """
    last_err: Optional[Exception] = None

    for attempt in range(1, RETRIES + 1):
        try:
            resp = session.get(url, timeout=TIMEOUT_SEC, allow_redirects=True)
            if resp.status_code != 200:
                # не падаем, просто считаем ошибкой
                return None

            # Важная проверка: страница должна быть HTML (текстовая)
            if not is_html_response(resp):
                return None

            # requests сам определяет кодировку; у Wikipedia обычно UTF-8
            return resp.text

        except (requests.RequestException, Exception) as e:
            last_err = e
            # небольшая пауза перед повтором
            time.sleep(0.5 * attempt)

    return None


# ----------------------------
# СОХРАНЕНИЕ: выкачка + index.txt
# ----------------------------
def save_page(output_dir: Path, number: int, html: str) -> Path:
    """
    Сохраняем страницу целиком (HTML с разметкой) в отдельный .txt файл.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    file_path = output_dir / f"{number}.txt"
    file_path.write_text(html, encoding="utf-8", errors="replace")
    return file_path


def write_index(index_path: Path, index_lines: List[Tuple[int, str]]) -> None:
    """
    Пишем index.txt:
    номер<TAB>URL
    """
    text = "\n".join(f"{n}\t{url}" for n, url in index_lines) + "\n"
    index_path.write_text(text, encoding="utf-8")


# ----------------------------
# MAIN: основная логика краулинга
# ----------------------------
def main() -> None:
    urls = load_urls(URLS_FILE)
    if not urls:
        raise SystemExit("Список URL пуст. Заполни urls.json.")

    session = create_http_session()
    out_dir = Path(OUTPUT_DIR)
    index_path = Path(INDEX_PATH)

    saved = 0
    index_lines: List[Tuple[int, str]] = []

    # Идём по списку, пока не сохраним MAX_PAGES успешных HTML-страниц
    for url in urls:
        if saved >= MAX_PAGES:
            break

        html = fetch_html(session, url)
        if html is None:
            print(f"[SKIP] {url}")
            time.sleep(REQUEST_DELAY_SEC)
            continue

        saved += 1
        save_page(out_dir, saved, html)
        index_lines.append((saved, url))
        print(f"[OK]   {saved:03d}: {url}")

        time.sleep(REQUEST_DELAY_SEC)

    write_index(index_path, index_lines)

    # Контроль по ТЗ
    if saved < MAX_PAGES:
        raise SystemExit(
            f"Не удалось сохранить 100 страниц: сохранено {saved}. "
            f"Добавь больше URL в urls.json (лучше 150–200) или проверь доступность сайтов."
        )

    print(f"Готово: сохранено {saved} страниц, index.txt сформирован.")


if __name__ == "__main__":
    main()
