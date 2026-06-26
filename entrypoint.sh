#!/bin/bash
set -e

# Activate virtual environment
export PATH="/app/.venv/bin:$PATH"

# Ensure matplotlib is configured to use a writeable temporary directory
export MPLCONFIGDIR="/tmp/matplotlib"

# If the command starts with an option (e.g. -f or --some-option), prepend "python"
if [ "${1#-}" != "$1" ]; then
    set -- python "$@"
fi

# Execute the command passed to the container
exec "$@"
