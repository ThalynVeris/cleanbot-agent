#!/usr/bin/env sh
set -eu

# Local API traffic must not pass through a system/VPN HTTP proxy.
API_BASE_URL="http://127.0.0.1:8000"
NO_PROXY="${NO_PROXY:+${NO_PROXY},}127.0.0.1,localhost"
no_proxy="${no_proxy:+${no_proxy},}127.0.0.1,localhost"
export API_BASE_URL NO_PROXY no_proxy

exec streamlit run app.py --server.port 8501
