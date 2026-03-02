"""
Модуль для булевого поиска по инвертированному индексу.

Поддерживает операторы: AND, OR, NOT
Поддерживает скобки для группировки выражений.
"""
from typing import Dict, Set, List, Union


# ----------------------------
# ПАРСЕР БУЛЕВЫХ ЗАПРОСОВ
# ----------------------------
class BooleanQueryParser:
    """
    Парсер и интерпретатор булевых запросов.
    
    Поддерживает:
    - Операторы: AND, OR, NOT
    - Скобки для группировки
    - Примеры: "algorithm", "algorithm AND search", "(algorithm OR search) AND NOT tree"
    """
    
    def __init__(self, inverted_index: Dict[str, Set[int]], all_doc_ids: Set[int]):
        """
        Инициализирует парсер булевых запросов.
        
        - inverted_index: инвертированный индекс (термин -> множество ID документов)
        - all_doc_ids: множество всех ID документов (используется для операции NOT)
        """
        self.index = inverted_index
        self.all_doc_ids = all_doc_ids
    
    def _get_term_docs(self, term: str) -> Set[int]:
        """
        Возвращает множество документов для термина.
        Если термина нет в индексе, возвращает пустое множество.
        """
        term = term.strip().lower()
        return self.index.get(term, set())
    
    def _tokenize_query(self, query: str) -> List[Union[str, Set[int]]]:
        """
        Разбивает запрос на токены (термины и операторы).
        
        Возвращает список, где элементы - либо строки-операторы ("AND", "OR", "NOT"),
        либо множества ID документов, либо скобки '(' и ')'.
        """
        # Удаляем лишние пробелы и нормализуем
        query = query.strip()
        
        # Разбиваем на токены: слова, операторы, скобки
        tokens = []
        current_token = ""
        
        i = 0
        while i < len(query):
            char = query[i]
            
            if char == '(':
                if current_token.strip():
                    tokens.append(current_token.strip())
                    current_token = ""
                tokens.append('(')
            elif char == ')':
                if current_token.strip():
                    tokens.append(current_token.strip())
                    current_token = ""
                tokens.append(')')
            elif char.isspace():
                if current_token.strip():
                    tokens.append(current_token.strip())
                    current_token = ""
            else:
                current_token += char
            
            i += 1
        
        if current_token.strip():
            tokens.append(current_token.strip())
        
        # Преобразуем токены в множества документов или операторы
        result = []
        for token in tokens:
            if token in ('(', ')'):
                result.append(token)
            elif token.upper() in ("AND", "OR", "NOT"):
                result.append(token.upper())
            else:
                # Это термин
                docs = self._get_term_docs(token)
                result.append(docs)
        
        return result
    
    def _evaluate_expression(self, tokens: List[Union[str, Set[int]]]) -> Set[int]:
        """
        Вычисляет значение булевого выражения.
        
        Использует алгоритм с учетом приоритета операторов:
        1. Скобки
        2. NOT (унарный)
        3. AND
        4. OR
        """
        if not tokens:
            return set()
        
        # Обрабатываем скобки рекурсивно
        tokens = self._process_parentheses(tokens)
        
        if not tokens:
            return set()
        
        # Если остался один элемент - возвращаем его
        if len(tokens) == 1:
            if isinstance(tokens[0], set):
                return tokens[0]
            return set()
        
        # Обрабатываем унарный NOT
        tokens = self._process_not(tokens)
        
        if not tokens:
            return set()
        
        if len(tokens) == 1:
            if isinstance(tokens[0], set):
                return tokens[0]
            return set()
        
        # Обрабатываем AND (более высокий приоритет)
        tokens = self._process_and(tokens)
        
        if not tokens:
            return set()
        
        if len(tokens) == 1:
            if isinstance(tokens[0], set):
                return tokens[0]
            return set()
        
        # Обрабатываем OR
        tokens = self._process_or(tokens)
        
        if not tokens:
            return set()
        
        # Должен остаться один элемент - результат
        if len(tokens) == 1 and isinstance(tokens[0], set):
            return tokens[0]
        
        return set()
    
    def _process_parentheses(self, tokens: List[Union[str, Set[int]]]) -> List[Union[str, Set[int]]]:
        """
        Обрабатывает скобки в выражении.
        """
        result = []
        i = 0
        
        while i < len(tokens):
            if tokens[i] == '(':
                # Находим соответствующую закрывающую скобку
                depth = 1
                j = i + 1
                while j < len(tokens) and depth > 0:
                    if tokens[j] == '(':
                        depth += 1
                    elif tokens[j] == ')':
                        depth -= 1
                    j += 1
                
                # Вычисляем выражение внутри скобок
                sub_expression = tokens[i+1:j-1]
                sub_result = self._evaluate_expression(sub_expression)
                result.append(sub_result)
                
                i = j
            else:
                result.append(tokens[i])
                i += 1
        
        return result
    
    def _process_not(self, tokens: List[Union[str, Set[int]]]) -> List[Union[str, Set[int]]]:
        """
        Обрабатывает унарный оператор NOT.
        """
        result = []
        i = 0
        
        while i < len(tokens):
            if tokens[i] == "NOT":
                # NOT применяется к следующему элементу
                if i + 1 < len(tokens):
                    if isinstance(tokens[i + 1], set):
                        # NOT множества = все документы минус это множество
                        result.append(self.all_doc_ids - tokens[i + 1])
                        i += 2
                    else:
                        # Если после NOT не множество, пропускаем NOT
                        i += 1
                else:
                    i += 1
            else:
                result.append(tokens[i])
                i += 1
        
        return result
    
    def _process_and(self, tokens: List[Union[str, Set[int]]]) -> List[Union[str, Set[int]]]:
        """
        Обрабатывает бинарный оператор AND.
        """
        result = []
        i = 0
        
        while i < len(tokens):
            if isinstance(tokens[i], set):
                result.append(tokens[i])
                i += 1
            elif tokens[i] == "AND":
                # AND между предыдущим и следующим элементом
                if result and isinstance(result[-1], set) and i + 1 < len(tokens) and isinstance(tokens[i + 1], set):
                    result[-1] = result[-1] & tokens[i + 1]
                    i += 2
                else:
                    # Некорректный запрос - пропускаем AND
                    i += 1
            else:
                result.append(tokens[i])
                i += 1
        
        return result
    
    def _process_or(self, tokens: List[Union[str, Set[int]]]) -> List[Union[str, Set[int]]]:
        """
        Обрабатывает бинарный оператор OR.
        """
        result = []
        i = 0
        
        while i < len(tokens):
            if isinstance(tokens[i], set):
                result.append(tokens[i])
                i += 1
            elif tokens[i] == "OR":
                # OR между предыдущим и следующим элементом
                if result and isinstance(result[-1], set) and i + 1 < len(tokens) and isinstance(tokens[i + 1], set):
                    result[-1] = result[-1] | tokens[i + 1]
                    i += 2
                else:
                    # Некорректный запрос - пропускаем OR
                    i += 1
            else:
                result.append(tokens[i])
                i += 1
        
        return result
    
    def search(self, query: str) -> Set[int]:
        """
        Выполняет булев поиск по запросу.
        
        Args:
            query: Булев запрос (например, "algorithm AND search", 
                   "(algorithm OR search) AND NOT tree")
        
        Returns:
            Множество ID документов, соответствующих запросу
        """
        if not query or not query.strip():
            return set()
        
        tokens = self._tokenize_query(query)
        return self._evaluate_expression(tokens)

