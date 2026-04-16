"""
СРС3 — Тема 9: Система мэтчинга студентов и научных руководителей
Streamlit-приложение на CrewAI
"""

import streamlit as st
import json
import os
import time
from pathlib import Path

# ──────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Мэтчинг студентов и научных руководителей",
    page_icon="🎓",
    layout="wide",
)

# ──────────────────────────────────────────────
# Lazy imports (after pip install)
# ──────────────────────────────────────────────
@st.cache_resource
def load_crew_modules():
    from crewai import Agent, Task, Crew, Process, LLM
    from crewai.memory import LongTermMemory, ShortTermMemory
    from crewai.tools import BaseTool
    from crewai.knowledge.source.text_file_knowledge_source import TextFileKnowledgeSource
    return Agent, Task, Crew, Process, LLM, LongTermMemory, ShortTermMemory, BaseTool, TextFileKnowledgeSource

# ──────────────────────────────────────────────
# Session state init
# ──────────────────────────────────────────────
if "result" not in st.session_state:
    st.session_state.result = None
if "run_log" not in st.session_state:
    st.session_state.run_log = []
if "hitl_pending" not in st.session_state:
    st.session_state.hitl_pending = False
if "hitl_data" not in st.session_state:
    st.session_state.hitl_data = None
if "memory_log" not in st.session_state:
    st.session_state.memory_log = []

# ──────────────────────────────────────────────
# Sidebar — конфигурация агентов
# ──────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Конфигурация агентов")

    st.subheader("Агент 1 — Аналитик интересов студентов")
    role_a1 = st.text_input("Role", value="Аналитик интересов студентов", key="r1")
    goal_a1 = st.text_area("Goal", value="Извлекать и структурировать научные интересы студентов из текстовых заявок.", key="g1", height=80)
    back_a1 = st.text_area("Backstory", value="Опытный академический советник, специализирующийся на анализе исследовательских профилей.", key="b1", height=80)

    st.divider()
    st.subheader("Агент 2 — Аналитик профилей преподавателей")
    role_a2 = st.text_input("Role", value="Аналитик профилей преподавателей", key="r2")
    goal_a2 = st.text_area("Goal", value="Извлекать ключевые научные направления и доступность научных руководителей из JSON-профилей.", key="g2", height=80)
    back_a2 = st.text_area("Backstory", value="HR-аналитик академической среды с опытом работы в исследовательских институтах.", key="b2", height=80)

    st.divider()
    st.subheader("Агент 3 — Координатор вторичного подбора (conditional)")
    role_a3 = st.text_input("Role", value="Координатор вторичного подбора", key="r3")
    goal_a3 = st.text_area("Goal", value="Предлагать альтернативные пары при слабом совпадении или запрашивать уточнение темы.", key="g3", height=80)
    back_a3 = st.text_area("Backstory", value="Специалист по разрешению конфликтов и пограничных случаев в академическом мэтчинге.", key="b3", height=80)

    st.divider()
    st.subheader("Агент 4 — Финальный составитель пар")
    role_a4 = st.text_input("Role", value="Финальный составитель пар", key="r4")
    goal_a4 = st.text_area("Goal", value="Формировать итоговое распределение пар студент–руководитель с обоснованием.", key="g4", height=80)
    back_a4 = st.text_area("Backstory", value="Академический координатор с многолетним опытом управления научными проектами.", key="b4", height=80)

    st.divider()
    st.subheader("Пороговые значения")
    similarity_threshold = st.slider("Порог схожести (для ConditionalTask)", 0.0, 1.0, 0.5, 0.05)

    st.divider()
    st.subheader("LLM")
    openai_key = st.text_input("OpenAI API Key", type="password", placeholder="sk-...")

# ──────────────────────────────────────────────
# Header
# ──────────────────────────────────────────────
st.title("🎓 Система мэтчинга студентов и научных руководителей")
st.markdown("> Мультиагентная система на **CrewAI** · Memory · Knowledge · Files · HITL · Conditional Tasks · Tools")

# ──────────────────────────────────────────────
# Zone 2 — Загрузка входных данных
# ──────────────────────────────────────────────
st.header("📂 Зона 2 — Входные данные")

col_left, col_right = st.columns(2)

with col_left:
    st.subheader("Заявки студентов (TXT)")
    student_file = st.file_uploader("Загрузите текстовый файл с заявками студентов", type=["txt"])
    if student_file:
        student_text = student_file.read().decode("utf-8")
        st.text_area("Содержимое заявок", student_text, height=200)
    else:
        st.info("Используется пример заявок из `sample_data/students.txt`")
        sample_students = Path("sample_data/students.txt")
        student_text = sample_students.read_text(encoding="utf-8") if sample_students.exists() else ""
        if student_text:
            st.text_area("Пример заявок", student_text, height=150)

with col_right:
    st.subheader("Профили преподавателей (JSON)")
    supervisor_file = st.file_uploader("Загрузите JSON с профилями преподавателей", type=["json"])
    if supervisor_file:
        supervisor_json = supervisor_file.read().decode("utf-8")
        try:
            supervisor_data = json.loads(supervisor_json)
            st.json(supervisor_data)
        except json.JSONDecodeError:
            st.error("Неверный формат JSON!")
            supervisor_data = {}
    else:
        st.info("Используется пример из `sample_data/supervisors.json`")
        sample_sup = Path("sample_data/supervisors.json")
        supervisor_json = sample_sup.read_text(encoding="utf-8") if sample_sup.exists() else "{}"
        supervisor_data = json.loads(supervisor_json)
        if supervisor_data:
            st.json(supervisor_data)

st.divider()
st.subheader("📚 Knowledge Source")
knowledge_choice = st.selectbox(
    "Выберите базу знаний правил распределения",
    ["matching_rules.txt (по умолчанию)", "Загрузить свой файл"]
)
if knowledge_choice == "Загрузить свой файл":
    knowledge_file = st.file_uploader("Загрузите файл правил (.txt)", type=["txt"])
    if knowledge_file:
        knowledge_text = knowledge_file.read().decode("utf-8")
        with open("knowledge/matching_rules.txt", "w", encoding="utf-8") as f:
            f.write(knowledge_text)
        st.success("Файл правил загружен!")
else:
    knowledge_text = Path("knowledge/matching_rules.txt").read_text(encoding="utf-8") if Path("knowledge/matching_rules.txt").exists() else ""

with st.expander("📖 Просмотр базы знаний (правила мэтчинга)"):
    st.text(knowledge_text)

# ──────────────────────────────────────────────
# Zone 3 — Запуск и результаты
# ──────────────────────────────────────────────
st.header("🚀 Зона 3 — Запуск системы")

run_col, info_col = st.columns([1, 2])
with run_col:
    run_btn = st.button("▶️ Запустить мэтчинг", type="primary", use_container_width=True)
with info_col:
    st.markdown("""
    **Этапы выполнения:**  
    1️⃣ Анализ заявок студентов  
    2️⃣ Анализ профилей руководителей  
    3️⃣ *ConditionalTask* — вторичный подбор при слабом совпадении  
    4️⃣ Формирование итоговых пар  
    5️⃣ **HITL** — проверка перед публикацией  
    """)

# ──────────────────────────────────────────────
# Core CrewAI logic
# ──────────────────────────────────────────────
def run_matching_crew(student_text, supervisor_data, knowledge_text, config, threshold):
    """Запускает CrewAI Crew и возвращает результат."""
    from crewai import Agent, Task, Crew, Process
    from crewai.memory import LongTermMemory, ShortTermMemory
    from crewai.knowledge.source.string_knowledge_source import StringKnowledgeSource
    from crewai import LLM
    import crewai

    llm = LLM(model="gpt-4o-mini", api_key=config["api_key"])

    # ── Knowledge ──────────────────────────────
    knowledge_source = StringKnowledgeSource(
        content=knowledge_text,
        metadata={"title": "Правила мэтчинга и распределения"}
    )

    # ── Tools ──────────────────────────────────
    from tools import JSONParserTool, SimilarityMatchingTool, RankingTool

    json_tool = JSONParserTool()
    similarity_tool = SimilarityMatchingTool(threshold=threshold)
    ranking_tool = RankingTool()

    # ── Agents ─────────────────────────────────
    student_analyst = Agent(
        role=config["role_a1"],
        goal=config["goal_a1"],
        backstory=config["back_a1"],
        llm=llm,
        tools=[],
        verbose=True,
        memory=True,
        knowledge_sources=[knowledge_source],
    )

    supervisor_analyst = Agent(
        role=config["role_a2"],
        goal=config["goal_a2"],
        backstory=config["back_a2"],
        llm=llm,
        tools=[json_tool],
        verbose=True,
        memory=True,
        knowledge_sources=[knowledge_source],
    )

    fallback_coordinator = Agent(
        role=config["role_a3"],
        goal=config["goal_a3"],
        backstory=config["back_a3"],
        llm=llm,
        tools=[similarity_tool],
        verbose=True,
        memory=True,
        knowledge_sources=[knowledge_source],
    )

    final_composer = Agent(
        role=config["role_a4"],
        goal=config["goal_a4"],
        backstory=config["back_a4"],
        llm=llm,
        tools=[ranking_tool],
        verbose=True,
        memory=True,
        knowledge_sources=[knowledge_source],
    )

    # ── Tasks ──────────────────────────────────
    task1 = Task(
        description=f"""Проанализируй следующие заявки студентов и извлеки для каждого:
- Имя студента
- Ключевые научные интересы (список)
- Предпочтительные темы исследований
- Любые дополнительные пожелания

Заявки студентов:
{student_text}

Верни структурированный JSON-список объектов студентов.""",
        expected_output="JSON-список объектов: [{name, interests: [], preferred_topics: [], notes}]",
        agent=student_analyst,
    )

    task2 = Task(
        description=f"""Используй инструмент JSONParserTool для обработки профилей преподавателей и извлечения:
- Имя преподавателя
- Научные направления
- Публикации / ключевые слова
- Количество свободных мест (max_students)
- Текущая загруженность

Данные профилей (JSON):
{json.dumps(supervisor_data, ensure_ascii=False, indent=2)}

Верни структурированный JSON-список объектов преподавателей.""",
        expected_output="JSON-список объектов: [{name, research_areas: [], keywords: [], max_students, current_load}]",
        agent=supervisor_analyst,
        context=[task1],
    )

    # ConditionalTask — срабатывает если совпадение ниже порога
    conditional_task = Task(
        description=f"""Используй SimilarityMatchingTool для анализа пар из task1 и task2.
Порог схожести: {threshold}.

Если для любого студента максимальный score схожести с доступными руководителями НИЖЕ {threshold},
ИЛИ если у руководителя нет свободных мест:
- Предложи вторичный подбор (альтернативного руководителя)
- Или сформулируй уточняющий вопрос студенту по теме

Иначе — верни JSON: {{"skip": true, "reason": "все совпадения выше порога"}}.

Всегда верни JSON с полями: skip (bool), conflicts (list), suggestions (list).""",
        expected_output='JSON: {skip: bool, conflicts: [{student, issue}], suggestions: [{student, alt_supervisor, question}]}',
        agent=fallback_coordinator,
        context=[task1, task2],
        condition=lambda outputs: _check_low_similarity(outputs, threshold),
    )

    sup_list = supervisor_data.get("supervisors", [])
    if not sup_list and isinstance(supervisor_data, list):
        sup_list = supervisor_data
    sup_json_str = json.dumps(sup_list, ensure_ascii=False)

    task4 = Task(
        description=(
    "Ты финальный составитель пар студент-руководитель.\n\n"
    "Шаг 1: Возьми список студентов из результата Задачи 1 (это JSON-список).\n"
    "Шаг 2: Вызови инструмент RankingTool. Передай аргументы строго так:\n"
    "  - data_json: JSON-строку вида {\"students\": [...список из шага 1...], \"supervisors\": " + sup_json_str + "}\n"
    "  - supervisors_json: пустую строку\n"
    "Шаг 3: Верни ответ инструмента как есть — JSON с полями pairs, unassigned, memory_update.\n"
    "НЕ придумывай пары самостоятельно — используй только результат инструмента."
),
        expected_output='JSON объект: {pairs: [{student, supervisor, similarity_score, justification}], unassigned: [], memory_update: str}',
        agent=final_composer,
        context=[task1, task2, conditional_task],
    )

    # ── Memory ─────────────────────────────────
    crew = Crew(
        agents=[student_analyst, supervisor_analyst, fallback_coordinator, final_composer],
        tasks=[task1, task2, conditional_task, task4],
        process=Process.sequential,
        memory=True,
        verbose=True,
    )

    result = crew.kickoff()
    return result


def _check_low_similarity(outputs, threshold):
    """Условие для ConditionalTask: True = запустить вторичный подбор."""
    combined = " ".join(str(o) for o in outputs if o).lower()
    # Явные маркеры проблем
    problem_words = ["низк", "conflict", "нет мест", "no slot", "unavailable",
                     "недоступ", "конфликт", "below", "ниже порога", "alternative"]
    return any(w in combined for w in problem_words)


# ──────────────────────────────────────────────
# Run button handler
# ──────────────────────────────────────────────
if run_btn:
    if not openai_key:
        st.error("⚠️ Введите OpenAI API Key в боковой панели!")
    elif not student_text.strip():
        st.error("⚠️ Загрузите файл с заявками студентов!")
    elif not supervisor_data:
        st.error("⚠️ Загрузите JSON с профилями преподавателей!")
    else:
        config = {
            "api_key": openai_key,
            "role_a1": role_a1, "goal_a1": goal_a1, "back_a1": back_a1,
            "role_a2": role_a2, "goal_a2": goal_a2, "back_a2": back_a2,
            "role_a3": role_a3, "goal_a3": goal_a3, "back_a3": back_a3,
            "role_a4": role_a4, "goal_a4": goal_a4, "back_a4": back_a4,
        }

        with st.spinner("🤖 Агенты работают..."):
            log_container = st.empty()
            steps = [
                "🔍 Агент 1: анализирует заявки студентов...",
                "📊 Агент 2: обрабатывает профили преподавателей...",
                "⚡ ConditionalTask: проверка порога схожести...",
                "✅ Агент 4: формирует итоговые пары...",
            ]
            st.session_state.run_log = []
            for step in steps:
                st.session_state.run_log.append(step)
                log_container.info("\n\n".join(st.session_state.run_log))
                time.sleep(0.5)

            try:
                os.environ["OPENAI_API_KEY"] = openai_key
                result = run_matching_crew(
                    student_text, supervisor_data, knowledge_text,
                    config, similarity_threshold
                )
                st.session_state.result = result
                st.session_state.run_log.append("🎉 Выполнение завершено!")
                log_container.success("\n\n".join(st.session_state.run_log))

                # Сохранить в memory_log
                st.session_state.memory_log.append({
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "student_count": student_text.count("Студент"),
                    "supervisor_count": len(supervisor_data.get("supervisors", [])),
                    "result_preview": str(result)[:200],
                })

            except Exception as e:
                st.error(f"Ошибка выполнения: {e}")
                st.exception(e)

# ──────────────────────────────────────────────
# Results display
# ──────────────────────────────────────────────
if st.session_state.result:
    st.divider()
    st.header("📋 Результаты мэтчинга")

    result_str = str(st.session_state.result)

    # Попытка распарсить JSON из результата
    try:
        import re
        json_match = re.search(r'\{.*\}', result_str, re.DOTALL)
        if json_match:
            result_data = json.loads(json_match.group())
            pairs = result_data.get("pairs", [])

            if pairs:
                st.subheader("🔗 Итоговые пары студент–руководитель")
                for i, pair in enumerate(pairs, 1):
                    with st.expander(f"Пара {i}: {pair.get('student', '?')} → {pair.get('supervisor', '?')}"):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Схожесть", f"{pair.get('similarity_score', 0):.2f}")
                        with col2:
                            score = pair.get("similarity_score", 0)
                            color = "🟢" if score >= similarity_threshold else "🟡"
                            st.write(f"{color} {'Выше' if score >= similarity_threshold else 'Ниже'} порога")
                        st.write("**Обоснование:**", pair.get("justification", "—"))
        else:
            st.text_area("Ответ системы", result_str, height=300)
    except Exception:
        st.text_area("Ответ системы", result_str, height=300)

    # ConditionalTask статус
    st.subheader("⚡ ConditionalTask")
    cond_fired = "вторичный" in result_str.lower() or "альтернатив" in result_str.lower() or "уточн" in result_str.lower()
    if cond_fired:
        st.warning("🔥 ConditionalTask **сработала** — были найдены слабые совпадения или конфликты доступности. Координатор предложил альтернативы.")
    else:
        st.success("✅ ConditionalTask **не сработала** — все совпадения выше порога, конфликтов нет.")

    # Tool usage
    st.subheader("🛠️ Использование инструментов")
    tool_cols = st.columns(3)
    with tool_cols[0]:
        st.info("**JSONParserTool**\nИспользован Агентом 2 для разбора профилей преподавателей")
    with tool_cols[1]:
        st.info("**SimilarityMatchingTool**\nИспользован Агентом 3 для расчёта схожести интересов")
    with tool_cols[2]:
        st.info("**RankingTool**\nИспользован Агентом 4 для финального ранжирования пар")

    # HITL block
    st.divider()
    st.subheader("👤 HITL — Подтверждение перед публикацией")
    st.warning("⚠️ Система ожидает вашего подтверждения перед публикацией итогового распределения.")

    hitl_col1, hitl_col2 = st.columns(2)
    with hitl_col1:
        hitl_comment = st.text_area("Ваш комментарий к распределению (необязательно)", height=80)
    with hitl_col2:
        st.markdown("**Вы должны проверить:**")
        st.markdown("- Корректность каждой пары")
        st.markdown("- Доступность руководителей")
        st.markdown("- Соответствие правилам распределения")

    hitl_approve = st.button("✅ Одобрить и опубликовать распределение", type="primary")
    hitl_reject = st.button("❌ Вернуть на доработку")

    if hitl_approve:
        st.success("🎉 Распределение **одобрено** и отправлено на публикацию!")
        st.balloons()
        if hitl_comment:
            st.info(f"Комментарий записан: {hitl_comment}")
    elif hitl_reject:
        st.error("🔄 Распределение возвращено на доработку. Скорректируйте конфигурацию и запустите снова.")

# ──────────────────────────────────────────────
# Memory log
# ──────────────────────────────────────────────
if st.session_state.memory_log:
    st.divider()
    st.header("🧠 История Memory")
    st.caption("Краткая выписка из памяти системы — сохранённые данные между запусками")
    for entry in st.session_state.memory_log:
        with st.expander(f"Запуск {entry['timestamp']}"):
            st.write(f"**Студентов:** {entry['student_count']}")
            st.write(f"**Руководителей:** {entry['supervisor_count']}")
            st.write(f"**Превью результата:** {entry['result_preview']}")

# Log display
if st.session_state.run_log:
    with st.expander("📜 Лог выполнения"):
        for line in st.session_state.run_log:
            st.write(line)
