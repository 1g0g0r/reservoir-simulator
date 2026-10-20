"""Точка входа интерактивной визуализации: `python visualise_solution.py`. Само приложение (Dash) - в
`paraphin/utils/visualisation_utils/visualisation.py`."""
import os
import shutil
import threading
import webbrowser

from paraphin.utils.visualisation_utils.visualisation import app, CACHE_DIR


def visualize_solution(port: int = 8050, debug: bool = False, open_browser: bool = True):
    """Запуск интерактивной визуализации; страница открывается в браузере, когда сервер уже слушает порт.

    Чтобы открыть с другого устройства, запустите с host='0.0.0.0', и на другом устройстве откройте http://<IP>:<порт>.
    Разрешите порт в брандмауэре. 0.0.0.0 открывает доступ всем в локальной сети.
    """
    # С debug перезагрузчик запускает модуль второй раз (WERKZEUG_RUN_MAIN) - вкладка открывается один раз
    if open_browser and os.environ.get('WERKZEUG_RUN_MAIN') != 'true':
        threading.Timer(1.0, webbrowser.open, args=(f'http://127.0.0.1:{port}/',)).start()
    try:
        app.run(host='127.0.0.1', port=port, debug=debug)
    finally:
        shutil.rmtree(CACHE_DIR, ignore_errors=True)


if __name__ == '__main__':
    visualize_solution()
