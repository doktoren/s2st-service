#!/bin/bash

sudo tailscale serve reset
sudo tailscale serve --bg --set-path=/ http://127.0.0.1:8002
sudo tailscale serve --bg --set-path=/translate http://127.0.0.1:8001/translate
sudo tailscale serve --bg --set-path=/ws http://127.0.0.1:8000/ws
