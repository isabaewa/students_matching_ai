"""
Кастомные инструменты (Tools) для мэтчинга студентов и научных руководителей.
Используются агентами CrewAI.
"""

import json
import re
from typing import Type, Optional, Any
from pydantic import BaseModel, Field

try:
    from crewai.tools import BaseTool
except ImportError:
    # Fallback для тестирования без установленного crewai
    class BaseTool:
        name: str = ""
        description: str = ""
        def _run(self, *args, **kwargs):
            raise NotImplementedError


# ──────────────────────────────────────────────
# Tool 1 — JSON Parser Tool
# ──────────────────────────────────────────────

class JSONParserInput(BaseModel):
    json_string: str = Field(description="JSON-строка для парсинга и структурирования")
    extract_keys: Optional[str] = Field(
        default="",
        description="Ключи для извлечения (через запятую). Если пусто — вернуть всё."
    )


class JSONParserTool(BaseTool):
    """
    Tool 1: Парсит JSON-профили преподавателей, извлекает и структурирует данные.
    Используется Агентом 2.
    """
    name: str = "JSONParserTool"
    description: str = (
        "Парсит JSON-строку с профилями преподавателей. "
        "Извлекает имена, научные направления, ключевые слова публикаций, "
        "доступность (max_students, current_load). "
        "Возвращает структурированный список профилей."
    )
    args_schema: Type[BaseModel] = JSONParserInput

    def _run(self, json_string: str, extract_keys: str = "") -> str:
        try:
            data = json.loads(json_string)
        except json.JSONDecodeError as e:
            return f"Ошибка парсинга JSON: {e}"

        supervisors = data.get("supervisors", data if isinstance(data, list) else [data])
        keys = [k.strip() for k in extract_keys.split(",")] if extract_keys else None

        result = []
        for sup in supervisors:
            if keys:
                entry = {k: sup.get(k) for k in keys if k in sup}
            else:
                entry = {
                    "name": sup.get("name", "Неизвестно"),
                    "research_areas": sup.get("research_areas", []),
                    "keywords": sup.get("keywords", []),
                    "publications": [p.get("title", "") for p in sup.get("publications", [])],
                    "max_students": sup.get("max_students", 1),
                    "current_load": sup.get("current_load", 0),
                    "available_slots": sup.get("max_students", 1) - sup.get("current_load", 0),
                }
            result.append(entry)

        return json.dumps(result, ensure_ascii=False, indent=2)


# ──────────────────────────────────────────────
# Tool 2 — Similarity Matching Tool
# ──────────────────────────────────────────────

class SimilarityInput(BaseModel):
    students_json: str = Field(description="JSON-список студентов с полем 'interests'")
    supervisors_json: str = Field(description="JSON-список преподавателей с полями 'research_areas' и 'keywords'")


class SimilarityMatchingTool(BaseTool):
    """
    Tool 2: Вычисляет схожесть интересов студентов и научных направлений преподавателей.
    Возвращает матрицу схожести и флаги конфликтов.
    Используется Агентом 3 (ConditionalTask).
    """
    name: str = "SimilarityMatchingTool"
    description: str = (
        "Вычисляет коэффициент схожести между интересами студентов и "
        "научными направлениями преподавателей на основе ключевых слов. "
        "Выявляет конфликты доступности. "
        "Возвращает матрицу схожести и список проблемных пар."
    )
    args_schema: Type[BaseModel] = SimilarityInput
    threshold: float = 0.5

    def _jaccard_similarity(self, set_a: set, set_b: set) -> float:
        """Коэффициент Жаккара как мера схожести множеств."""
        if not set_a or not set_b:
            return 0.0
        intersection = set_a & set_b
        union = set_a | set_b
        return len(intersection) / len(union)

    def _tokenize(self, text_or_list) -> set:
        """Токенизация: строку или список → множество слов в нижнем регистре."""
        if isinstance(text_or_list, list):
            text = " ".join(str(x) for x in text_or_list)
        else:
            text = str(text_or_list)
        tokens = re.findall(r'\b\w{3,}\b', text.lower())
        return set(tokens)

    def _run(self, students_json: str, supervisors_json: str) -> str:
        try:
            students = json.loads(students_json)
            supervisors = json.loads(supervisors_json)
        except json.JSONDecodeError as e:
            return f"Ошибка парсинга: {e}"

        matrix = []
        conflicts = []
        below_threshold = []

        for student in students:
            s_name = student.get("name", "Студент")
            s_interests = self._tokenize(
                student.get("interests", []) + student.get("preferred_topics", [])
            )

            best_score = 0.0
            best_sup = None
            row = {"student": s_name, "scores": {}}

            for sup in supervisors:
                sup_name = sup.get("name", "Руководитель")
                sup_keywords = self._tokenize(
                    sup.get("research_areas", []) + sup.get("keywords", [])
                )
                score = self._jaccard_similarity(s_interests, sup_keywords)
                available = sup.get("available_slots", 1)

                row["scores"][sup_name] = {
                    "score": round(score, 3),
                    "available": available > 0,
                }

                if available > 0 and score > best_score:
                    best_score = score
                    best_sup = sup_name

                if available <= 0:
                    conflicts.append({
                        "student": s_name,
                        "supervisor": sup_name,
                        "issue": "Нет свободных мест"
                    })

            row["best_match"] = best_sup
            row["best_score"] = round(best_score, 3)

            if best_score < self.threshold:
                below_threshold.append({
                    "student": s_name,
                    "best_score": round(best_score, 3),
                    "threshold": self.threshold,
                    "issue": "Совпадение ниже порога — требуется уточнение темы"
                })

            matrix.append(row)

        result = {
            "similarity_matrix": matrix,
            "conflicts": conflicts + below_threshold,
            "needs_fallback": len(below_threshold) > 0 or len(conflicts) > 0,
            "summary": f"Проверено студентов: {len(students)}, "
                       f"конфликтов: {len(conflicts)}, "
                       f"ниже порога: {len(below_threshold)}"
        }

        return json.dumps(result, ensure_ascii=False, indent=2)


# ──────────────────────────────────────────────
# Tool 3 — Ranking Tool
# ──────────────────────────────────────────────

class RankingInput(BaseModel):
    data_json: str = Field(
        description=(
            "JSON с данными для мэтчинга. Принимает ЛЮБОЙ из форматов:\n"
            "1) {\"similarity_matrix\": [...]} — готовая матрица от SimilarityMatchingTool\n"
            "2) {\"students\": [...], \"supervisors\": [...]} — сырые данные (инструмент сам считает схожесть)\n"
            "3) Просто список студентов [...] с полями name, interests, preferred_topics — "
            "тогда нужно также передать supervisors_json"
        )
    )
    supervisors_json: Optional[str] = Field(
        default="",
        description="JSON-список руководителей (нужен только если data_json не содержит supervisors)"
    )


class RankingTool(BaseTool):
    """
    Tool 3: Ранжирует и формирует оптимальное назначение пар студент–руководитель.
    Принимает данные в любом формате — сам нормализует и считает схожесть если нужно.
    Используется Агентом 4.
    """
    name: str = "RankingTool"
    description: str = (
        "Формирует финальное распределение пар студент–руководитель. "
        "Передай в data_json либо готовую матрицу схожести от SimilarityMatchingTool, "
        "либо объект с полями students и supervisors — инструмент сам вычислит схожесть. "
        "Учитывает ограничения по числу мест (max_students, current_load). "
        "Возвращает JSON: {pairs: [{student, supervisor, similarity_score, justification}], unassigned, memory_update}."
    )
    args_schema: Type[BaseModel] = RankingInput

    # ── helpers ──────────────────────────────────────────────────
    def _tokenize(self, val) -> set:
        if isinstance(val, list):
            text = " ".join(str(x) for x in val)
        else:
            text = str(val or "")
        return set(re.findall(r'\b\w{3,}\b', text.lower()))

    def _jaccard(self, a: set, b: set) -> float:
        if not a or not b:
            return 0.0
        return len(a & b) / len(a | b)

    def _build_matrix(self, students: list, supervisors: list) -> list:
        """Строит матрицу схожести из сырых данных."""
        matrix = []
        for s in students:
            s_name = s.get("name", "Студент")
            s_tokens = self._tokenize(
                s.get("interests", []) + s.get("preferred_topics", [])
            )
            row = {"student": s_name, "scores": {}}
            for sup in supervisors:
                sup_name = sup.get("name", "Руководитель")
                sup_tokens = self._tokenize(
                    sup.get("research_areas", []) + sup.get("keywords", [])
                )
                score = self._jaccard(s_tokens, sup_tokens)
                max_s = sup.get("max_students", 3)
                load = sup.get("current_load", 0)
                slots = sup.get("available_slots", max_s - load)
                row["scores"][sup_name] = {
                    "score": round(score, 3),
                    "available": slots > 0,
                    "max_students": max_s,
                }
            matrix.append(row)
        return matrix

    def _run(self, data_json: str, supervisors_json: str = "") -> str:
        # ── 1. Parse input ────────────────────────────────────────
        try:
            data = json.loads(data_json)
        except json.JSONDecodeError as e:
            return json.dumps({"error": f"Ошибка парсинга data_json: {e}"}, ensure_ascii=False)

        supervisors_extra = []
        if supervisors_json and supervisors_json.strip():
            try:
                sv = json.loads(supervisors_json)
                if isinstance(sv, dict):
                    supervisors_extra = sv.get("supervisors", [sv])
                elif isinstance(sv, list):
                    supervisors_extra = sv
            except json.JSONDecodeError:
                pass

        # ── 2. Normalise to similarity_matrix ─────────────────────
        if isinstance(data, list):
            # Список студентов — нужны руководители
            if not supervisors_extra:
                return json.dumps({
                    "error": "Передан список студентов, но supervisors_json пуст. "
                             "Передай объект {students:[...], supervisors:[...]}."
                }, ensure_ascii=False)
            matrix = self._build_matrix(data, supervisors_extra)

        elif isinstance(data, dict):
            if "similarity_matrix" in data:
                # Готовая матрица от SimilarityMatchingTool
                matrix = data["similarity_matrix"]
            elif "students" in data and "supervisors" in data:
                # Сырые данные в одном объекте
                matrix = self._build_matrix(data["students"], data["supervisors"])
            elif "students" in data and supervisors_extra:
                matrix = self._build_matrix(data["students"], supervisors_extra)
            elif "pairs" in data:
                # Уже готовые пары — просто вернуть
                return json.dumps(data, ensure_ascii=False, indent=2)
            else:
                # Последняя попытка: может быть одиночный студент
                return json.dumps({
                    "error": "Не удалось определить формат данных. "
                             "Передай {students:[...], supervisors:[...]} или матрицу схожести.",
                    "received_keys": list(data.keys()),
                }, ensure_ascii=False)
        else:
            return json.dumps({"error": f"Неожиданный тип данных: {type(data)}"}, ensure_ascii=False)

        # ── 3. Greedy matching ────────────────────────────────────
        supervisor_slots: dict = {}  # sup_name -> count assigned

        # Собрать все кандидатов (student, supervisor, score)
        candidates = []
        for row in matrix:
            student = row.get("student", "")
            for sup_name, info in row.get("scores", {}).items():
                if info.get("available", True):
                    candidates.append({
                        "student": student,
                        "supervisor": sup_name,
                        "score": info.get("score", 0.0),
                        "max_students": info.get("max_students", 3),
                    })

        # Сортировка по убыванию score
        candidates.sort(key=lambda x: x["score"], reverse=True)

        assigned_students: set = set()
        pairs = []

        for c in candidates:
            s = c["student"]
            sup = c["supervisor"]
            max_s = c.get("max_students", 3)

            if s in assigned_students:
                continue
            if supervisor_slots.get(sup, 0) >= max_s:
                continue

            score = c["score"]
            pairs.append({
                "student": s,
                "supervisor": sup,
                "similarity_score": score,
                "justification": (
                    f"Коэффициент схожести: {score:.2f}. "
                    f"Интересы студента наилучшим образом совпадают с направлениями руководителя "
                    f"среди всех доступных вариантов."
                ),
            })
            assigned_students.add(s)
            supervisor_slots[sup] = supervisor_slots.get(sup, 0) + 1

        all_students = [r.get("student", "") for r in matrix]
        unassigned = [s for s in all_students if s not in assigned_students]

        memory_update = (
            f"Назначено {len(pairs)} пар: "
            + ", ".join(f"{p['student']} → {p['supervisor']}" for p in pairs)
            + (f". Без назначения: {unassigned}" if unassigned else ".")
        )

        return json.dumps({
            "pairs": pairs,
            "unassigned": unassigned,
            "memory_update": memory_update,
        }, ensure_ascii=False, indent=2)
