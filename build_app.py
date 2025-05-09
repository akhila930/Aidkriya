import PyInstaller.__main__
import os

# Get the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))

PyInstaller.__main__.run([
    'aidkriya_prediction_app.py',
    '--onefile',
    '--windowed',
    '--name=AidkriyaDonationPredictor',
    f'--add-data={os.path.join(current_dir, "Year-important-dates-data(Sheet1) (1).csv")};.',
    '--icon=NONE'
])
