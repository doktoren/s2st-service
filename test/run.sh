#!/bin/bash

echo "Go to http://localhost:8001/"
exec python -m http.server 8001
