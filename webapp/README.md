# Azure Startup Command
gunicorn --bind=0.0.0.0 --timeout 600 --workers 2 --threads 2 app:app