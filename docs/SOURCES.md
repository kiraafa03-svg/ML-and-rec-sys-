# Проверенные источники

Проверка доступности и содержания: 25.10.2026.

1. [Kaggle competition data](https://www.kaggle.com/competitions/sweettv-tv-program-recommender/data).
   Источник данных и описание формирования целей. Полный live доступ к файлам
   требует своего аккаунта и принятия Rules. Локальные ничьи разрешаются по ID;
   правило host для ничьих не установлено.
2. Y. Hu, Y. Koren, C. Volinsky, **Collaborative Filtering for Implicit Feedback Datasets**,
   ICDM 2008, DOI [10.1109/ICDM.2008.22](https://doi.org/10.1109/ICDM.2008.22).
   [PDF на сайте автора](https://yifanhu.net/PUB/cf.pdf) прочитан для разделения
   implicit preference/confidence и ALS objective. Вариант log target обозначен
   как собственная модификация, а не исходная бинарная модель статьи.
3. [NetworkX pagerank](https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.link_analysis.pagerank_alg.pagerank.html).
   Официальное описание персонализации и dangling distribution. NetworkX
   используется как источник спецификации, а не библиотека вычислений проекта.
4. [SciPy sparse documentation](https://docs.scipy.org/doc/scipy/reference/sparse.html).
   Официальная документация используемых разреженных структур.
5. [Решение-пример, закреплённый commit](https://github.com/redlymood/Machine-Learning-and-Recommender-Systems/tree/e7895a18e57d8111456cead185a0216d3eec9d93/exam).
   Внешний материал для аудита и организации проекта. Его результаты не являются
   экспериментами этого репозитория.
6. [Kaggle CLI commands](https://github.com/Kaggle/kaggle-cli/blob/main/docs/competitions.md).
   Download, submit, submission status, submissions CSV и submission-download.
   Синтаксис проверен также по `--help` установленного Kaggle CLI 2.2.5.
7. [Codex Local/Cloud](https://learn.chatgpt.com/docs/environments/modes) и
   [машинные события CLI](https://learn.chatgpt.com/docs/non-interactive-mode).
   Источники ограничений среды и работы с исходным журналом.

Шаблон отчёта: приложенный пользователем «Пример отчета .pdf». Его структура
использована для генератора; текст, эксперименты и личные проверки автора
образца не копируются в отчёт Кирилла Афанасьева.
