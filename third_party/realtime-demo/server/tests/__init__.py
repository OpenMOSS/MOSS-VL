# Hermeticity: every `python -m server.tests.X` run imports this first — a
# box-local .env.deploy must never leak into tests (config layer 3 off).
# setdefault, so a developer can still opt in with an explicit ENV_DEPLOY_FILE.
import os

os.environ.setdefault("ENV_DEPLOY_FILE", "")
