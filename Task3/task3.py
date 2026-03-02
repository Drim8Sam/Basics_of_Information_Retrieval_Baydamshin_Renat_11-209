"""
Задание 3: Инвертированный индекс и булев поиск.

1. Создает инвертированный индекс из токенов документов
2. Реализует булев поиск с операторами AND, OR, NOT
3. Поддерживает сложные запросы со скобками
"""
from pathlib import Path
from typing import Set

from inverted_index import build_inverted_index, save_inverted_index, load_inverted_index
from boolean_search import BooleanQueryParser

TOKENS_DIR_RELATIVE = "Task2/tokens_per_doc"
INDEX_FILE_RELATIVE = "Task3/inverted_index.txt"

def get_all_doc_ids(tokens_dir: Path) -> Set[int]:
    """
    Получает множество всех ID документов из файлов токенов.
    """
    doc_ids = set()
    
    for token_file in tokens_dir.glob("*_tokens.txt"):
        try:
            doc_id = int(token_file.stem.replace("_tokens", ""))
            doc_ids.add(doc_id)
        except ValueError:
            continue
    
    return doc_ids

def main() -> None:
    """
    Основная функция для запуска индексации и поиска.
    """
    project_root = Path(__file__).resolve().parents[1]
    tokens_dir = project_root / TOKENS_DIR_RELATIVE
    index_file = project_root / INDEX_FILE_RELATIVE
    
    if not tokens_dir.exists():
        raise SystemExit(
            f"Ошибка: директория {tokens_dir} не найдена.\n"
            "Сначала выполните Task2 для создания токенов."
        )
    
    print("=" * 60)
    print("Задание 3: Инвертированный индекс и булев поиск")
    print("=" * 60)
    
    # Строим или загружаем индекс
    print("\n1. Построение инвертированного индекса...")
    
    if index_file.exists():
        print(f"   Загружаем индекс из {index_file.name}")
        inverted_index = load_inverted_index(index_file)
        print(f"   Загружено {len(inverted_index)} терминов")
    else:
        print(f"   Строим индекс из файлов в {tokens_dir.name}/")
        inverted_index = build_inverted_index(tokens_dir)
        print(f"   Построено {len(inverted_index)} терминов")
        
        # Сохраняем индекс
        print(f"   Сохраняем индекс в {index_file.name}")
        save_inverted_index(inverted_index, index_file)
    
    # Получаем множество всех документов
    all_doc_ids = get_all_doc_ids(tokens_dir)
    print(f"   Всего документов: {len(all_doc_ids)}")
    
    # Создаем парсер запросов
    parser = BooleanQueryParser(inverted_index, all_doc_ids)
    
    print("\n2. Булев поиск готов к использованию")
    print("\n" + "=" * 60)
    print("Инструкция по использованию:")
    print("  - Операторы: AND, OR, NOT")
    print("  - Можно использовать скобки: (термин1 AND термин2) OR термин3")
    print("  - Примеры запросов:")
    print("    * algorithm")
    print("    * algorithm AND search")
    print("    * algorithm OR search")
    print("    * algorithm AND NOT tree")
    print("    * (algorithm OR search) AND NOT tree")
    print("  - Введите 'exit' или 'quit' для выхода")
    print("=" * 60 + "\n")
    
    # Цикл ввода запросов
    while True:
        try:
            query = input("Введите запрос: ").strip()
            
            if not query:
                continue
            
            if query.lower() in ('exit', 'quit', 'q'):
                print("Выход из программы.")
                break
            
            # Выполняем поиск
            result_docs = parser.search(query)
            
            # Выводим результаты
            print(f"\nРезультат поиска: найдено {len(result_docs)} документов")
            
            if result_docs:
                sorted_docs = sorted(result_docs)
                print("ID документов:", ", ".join(map(str, sorted_docs[:50])))
                if len(sorted_docs) > 50:
                    print(f"... и еще {len(sorted_docs) - 50} документов")
            else:
                print("Документы не найдены.")
            
            print()
            
        except KeyboardInterrupt:
            print("\n\nВыход из программы.")
            break
        except Exception as e:
            print(f"\nОшибка при обработке запроса: {e}")
            print("Попробуйте другой запрос.\n")


if __name__ == "__main__":
    main()

