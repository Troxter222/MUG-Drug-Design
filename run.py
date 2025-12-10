import os
import sys

# Добавляем корневую папку в путь, чтобы импорты работали
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.bot import bot

if __name__ == "__main__":
    print("🚀 Launching Molecular Universe Generator...")
    try:
        bot.polling(none_stop=True)
    except Exception as e:
        print(f"CRITICAL ERROR: {e}")