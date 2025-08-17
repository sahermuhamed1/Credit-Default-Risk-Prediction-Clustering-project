import sys
import os

# Add the project root to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = current_dir
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from app import create_app

app = create_app()

if __name__ == '__main__':
    # Use reloader_interval to make sure reloader is responsive,
    # or set use_reloader=False if it causes issues.
    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=True, reloader_interval=1)