# Машинное обучение и рекомендательные системы

Индивидуальный экзаменационный проект: **sweet.tv — TV program recommender**.
Студент: **Кирилл Афанасьев**. Курс: LLM4Rec, AVITO.

## Текущий статус

Реализованы подготовка данных, popularity baseline, Personalized PageRank,
матричная факторизация для implicit feedback, гибрид, поиск 150 конфигураций,
проверка submission и генератор PDF по предоставленному образцу.

**Реальные данные Kaggle ещё не предоставлены. Реальные 150 экспериментов,
private score и финальный экзаменационный PDF пока не получены.**
Проверки программного кода включают полный цикл на маленьком **синтетическом**
наборе. Они не являются результатами соревнования. Актуальная запись проверок:
[`exam/results/software_checks.json`](exam/results/software_checks.json).

Исходный журнал этой облачной AI-сессии недоступен из рабочей среды.
Он не заменён реконструкцией или JSON со ссылкой. Для сдачи потребуется
оригинальный экспорт с запросами, кодом, вызовами инструментов и их результатами.

## Структура

| Путь | Содержание |
|---|---|
| `exam/data.py` | Потоковое чтение CSV, временные границы, interval join и кеш |
| `exam/models.py` | PPR, weighted ALS, fusion, общий fallback и Top-5 |
| `exam/metrics.py` | AP@5 и macro MAP@5 |
| `exam/experiments.py` | Фиксированный план 150 экспериментов, resume, выбор, refit |
| `exam/validation.py` | Независимый расчёт AP и проверка CSV/экспериментов |
| `exam/report.py` | PDF по структуре приложенного образца |
| `exam/session.py` | Проверка структуры и побайтовое копирование существующего экспорта |
| `exam/tests/` | Независимые численные проверки и регрессионные тесты |
| `exam/results/` | Проверки ПО; после реального запуска — все эксперименты и метрики |
| `exam/submissions/` | После реального запуска — `final.csv` |
| `scripts/run_local.py` | Подготовка → 150 экспериментов → refit → проверка |
| `scripts/record_kaggle_score.py` | Получение private score и сверка скачанного submission |
| `docs/REFERENCE_AUDIT.md` | Ошибки примера, контрпримеры и исправления |
| `docs/LOCAL_CODEX.md` | Локальный запуск и сохранение исходного журнала |
| `docs/PERSONAL_VERIFICATION.md` | Что студенту проверить самому |

## Установка и проверки

Python 3.11–3.13, CPU. GPU и TensorFlow не требуются.
В команде `python` используйте интерпретатор созданного окружения.

```bash
python -m venv .venv
# macOS/Linux:
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python scripts/check_project.py
```

Проверяются точный interval join против перебора, half-open границы,
объединение дублей, отсутствие временной утечки, user ID вплоть до uint64 max,
AP против независимого оракула, PPR с dangling node против решения системы,
ALS против прямой функции потерь и нормальных уравнений, воспроизводимость,
resume, все 150 конфигураций на синтетике и два побайтово одинаковых CSV.

## Данные Kaggle

Соревнование:
https://www.kaggle.com/competitions/sweettv-tv-program-recommender

Сначала примите Rules своим аккаунтом. Установите Kaggle CLI в том же
локальном окружении и выполните штатную авторизацию:

```bash
python -m pip install -r requirements-kaggle.txt
kaggle auth login
kaggle competitions download sweettv-tv-program-recommender -p exam/data --unzip
python -m exam.cli preflight --data-dir exam/data
```

Принимаются исходные пять CSV, перечисленные в `exam/data/README.md`.
Число строк проверяется против экзаменационного задания. Raw CSV, токены,
кеши и личные журналы не включаются в публичный репозиторий.

## Реальные 150 экспериментов

```bash
python scripts/run_local.py --data-dir exam/data
```

Команда готовит данные, выполняет **40 PPR + 70 MF + 40 hybrid** конфигураций,
переобучает выбранную стратегию на полном train и проверяет `final.csv`.
После прерывания повторный запуск продолжает зафиксированный план.
Изменение кода, исходных данных или настроек требует нового каталога результатов.

Отдельные этапы:

```bash
python -m exam.cli prepare --data-dir exam/data --cache-dir exam/cache
python -m exam.cli tune --cache-dir exam/cache --output-dir exam/results
python -m exam.cli tune --cache-dir exam/cache --output-dir exam/results --resume
python -m exam.cli predict --cache-dir exam/cache --results-dir exam/results
python -m exam.cli verify --cache-dir exam/cache --results-dir exam/results --submission exam/submissions/final.csv
```

Для первичной проверки на реальных данных можно выполнить `tune --limit 2`,
затем продолжить с `--resume`. Финальный CSV блокируется до завершения всех 150.
Сбой не считается успешным экспериментом; запись ошибки сохраняется и повторяется
при resume. Baseline и регрессионные тесты считаются отдельно от 150 конфигураций.

## Протокол оценки

- Train: недели 11–20; validation: 21–30; final train: 11–30; test: 31–42.
- Просмотры обрезаются по границам **до** join; используются интервалы `[start, stop)`.
- Совпадение: тот же канал и положительное пересечение с эфиром.
- Просмотр засчитывается при `overlap / programme_duration >= 0.8`.
- `duraton` из лога — длина сессии, она не используется как длина программы.
- По умолчанию пересекающиеся интервалы одного пользователя на одном канале
  объединяются (`union`). Доступна явно задокументированная политика `sum`.
  Она меняет разметку; результаты разных политик нельзя смешивать в одном поиске.
- `tv_show_id = 0` не является item. Неразмеченное эфирное время может дать
  только исторический контекст каналов.
- Кандидаты — размеченные программы, выходящие в целевом периоде.
- Повторные рекомендации просмотренных программ разрешены для всех методов.
- Cold start: популярность в train с ограничением целевым каталогом;
  нулевая популярность разрешается по ID. Всегда пять уникальных ненулевых ID.
- Validation target: пять наиболее частых qualifying программ пользователя;
  основная когорта — пользователи минимум с пятью такими программами.
  Размер когорты, исключения, cold start и пограничные ничьи публикуются.
- AP@5 не начисляет повторный hit за дубли; MAP@5 — среднее по общей когорте.
- Правило host для ничьих не опубликовано: локально используется меньший ID.
- Все 150 конфигураций используют один holdout для **выбора**. Максимум на нём
  имеет selection bias. Private score Kaggle — отдельный внешний результат.
- Из будущего расписания используются наличие программы, канал и время эфира;
  будущее поведение пользователей и будущие контентные признаки не используются.

План параметров можно посмотреть до запуска:

```bash
python -m exam.cli plan
```

PPR: alpha × raw/log × recency × graph top-k; MF: factors × confidence × L2,
binary/log target; hybrid: вес MF × schedule exponent × min-max/RRF.
MF seed — 42. Численные значения и время каждого запуска сохраняются.
PPR сообщает residual и convergence; достижение лимита итераций не считается
сходимостью. ALS реализует implicit weighted least squares, а не модель рейтингов.

## Kaggle и подтверждение результата

После выбора и проверки CSV отправляется один designated final submission:

```bash
kaggle competitions submit sweettv-tv-program-recommender -f exam/submissions/final.csv -m "Validation-selected final; 150 recorded local configurations"
kaggle competitions submission SUBMISSION_REF
```

После COMPLETE сохраните **реальный** скриншот с private score и ref, затем:

```bash
python scripts/record_kaggle_score.py --submission-ref SUBMISSION_REF --screenshot private/kaggle_score.png
```

Скрипт читает private score из авторизованного ответа Kaggle, скачивает отправленный
CSV и сверяет его SHA-256 с локальным файлом. Оценка вручную не вводится.
Текущие команды CLI проверены на Kaggle CLI 2.2.5; live операции требуют
аккаунта, принятия Rules и доступных данных. Leaderboard заморожен; его позиция
не используется в отчёте.

## PDF и исходный журнал

```bash
python -m pip install -r requirements-report.txt
python -m exam.cli report --evidence exam/results/kaggle_evidence.json --session-file private/original_session.jsonl --personal-verification private/my_check.txt --email YOUR_COURSE_EMAIL
```

Генератор проверяет 150 экспериментов, соответствие CSV, private score и скриншот,
наличие исходного журнала и личной проверки. Он не создаёт итоговый отчёт из
синтетических результатов. Структура: title block, Abstract, Introduction,
Related Work, Method, Experiments, Discussion с двумя одинаковыми пользователями
для всех методов, AI Usage Disclosure, References.

Unicode serif font ищется в стандартных каталогах Linux/macOS/Windows;
при необходимости задайте `REPORT_FONT_REGULAR` и `REPORT_FONT_BOLD` путями к TTF.
После генерации обязательно откройте PDF и проверьте все страницы.

Исходный экспорт должен быть реальной записью работы, а не `results.json`,
сводкой действий или JSON с одной ссылкой. Инструмент `copy_native_session.py`
только копирует **уже существующий** экспорт побайтово и проверяет наличие
распознаваемых tool calls/results; он не доказывает происхождение файла.
Подробности — в `docs/LOCAL_CODEX.md`.

## Внешние материалы и AI assistance

Решение-пример изучено на commit
`e7895a18e57d8111456cead185a0216d3eec9d93`:
https://github.com/redlymood/Machine-Learning-and-Recommender-Systems/tree/e7895a18e57d8111456cead185a0216d3eec9d93/exam

Его private score, submissions и готовые результаты не перенесены в этот проект.
Реализация написана независимо с помощью OpenAI Codex; математические проверки
описаны в тестах, а граница личной проверки студента — отдельно.
Первичные источники алгоритмов и инструментов перечислены в `docs/SOURCES.md`.
