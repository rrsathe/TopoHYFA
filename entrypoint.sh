#!/bin/bash
set -e

# Activate virtual environment
export PATH="/app/.venv/bin:$PATH"

# Ensure matplotlib is configured to use a writeable temporary directory
export MPLCONFIGDIR="/tmp/matplotlib"

# Handle version queries
if [ "$1" = "version" ] || [ "$1" = "--version" ] || [ "$1" = "versions" ] || [ "$1" = "--versions" ]; then
    cat /app/package_versions.txt
    exit 0
fi

# If the command starts with an option (e.g. -f or --some-option), prepend "python"
if [ "${1#-}" != "$1" ]; then
    set -- python "$@"
fi

# Execute the command passed to the container
exec "$@"
