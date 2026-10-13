"""Run every test against a throwaway app folder, never the user's real %APPDATA%\\AlphaGo.

`alphago.config` loads settings.json at import time, so ALPHAGO_HOME must be set before anything imports alphago.
"""

import os
import tempfile

os.environ["ALPHAGO_HOME"] = tempfile.mkdtemp(prefix="alphago-tests-")
