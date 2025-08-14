#!/bin/bash

echo "Go to http://localhost:8002/"
exec python -m http.server 8002
