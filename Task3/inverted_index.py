"""
Модуль для построения инвертированного индекса.

Инвертированный индекс: термин -> [список ID документов, где встречается термин]
"""
from pathlib import Path
from typing import Dict, Set


def build_inverted_index(tokens_dir: Path) -> Dict[str, Set[int]]:
    """
    Строит инвертированный индекс из файлов токенов.
    
    Читает все файлы формата {id}_tokens.txt из указанной директории
    и создает структуру данных: термин -> множество ID документов.
    """
    inverted_index: Dict[str, Set[int]] = {}
    
    # Проходим по всем файлам токенов
    for token_file in sorted(tokens_dir.glob("*_tokens.txt")):
        # Извлекаем ID документа из имени файла (например, "1_tokens.txt" -> 1)
        try:
            doc_id = int(token_file.stem.replace("_tokens", ""))
        except ValueError:
            continue
        
        # Читаем токены из файла
        try:
            tokens = token_file.read_text(encoding="utf-8").strip().split("\n")
        except Exception:
            continue
        
        # Добавляем каждый токен в индекс
        for token in tokens:
            token = token.strip().lower()
            if token:  # Пропускаем пустые строки
                if token not in inverted_index:
                    inverted_index[token] = set()
                inverted_index[token].add(doc_id)
    
    return inverted_index

def save_inverted_index(index: Dict[str, Set[int]], output_path: Path) -> None:
    """
    Сохраняет инвертированный индекс в файл.
    
    Формат: каждая строка - "термин doc_id1 doc_id2 ... doc_idN"
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    lines = []
    for term in sorted(index.keys()):
        doc_ids = sorted(index[term])
        line = f"{term} {' '.join(map(str, doc_ids))}"
        lines.append(line)
    
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_inverted_index(index_path: Path) -> Dict[str, Set[int]]:
    """
    Загружает инвертированный индекс из файла.
    
    Формат: каждая строка - "термин doc_id1 doc_id2 ... doc_idN"
    """
    index: Dict[str, Set[int]] = {}
    
    if not index_path.exists():
        return index
    
    for line in index_path.read_text(encoding="utf-8").strip().split("\n"):
        if not line.strip():
            continue
        
        parts = line.split()
        if len(parts) < 2:
            continue
        
        term = parts[0].lower()
        doc_ids = {int(doc_id) for doc_id in parts[1:] if doc_id.isdigit()}
        if doc_ids:
            index[term] = doc_ids
    
    return index

