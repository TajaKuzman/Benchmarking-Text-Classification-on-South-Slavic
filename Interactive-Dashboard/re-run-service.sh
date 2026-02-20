#!/bin/bash
# A script that terminates any existing running service, reloads the system and runs it

# To be executed from the initial working directory (`\`), without any conda environment enabled

# Stop and disable any previously running service
systemctl --user stop interactive_dashboard
systemctl --user disable interactive_dashboard.service

echo "Any previously running service disabled."

# Reload the service
systemctl --user daemon-reload

# Enable the service
systemctl --user enable interactive_dashboard.service

# Start the service
systemctl --user start interactive_dashboard.service

echo "Service enabled and started."

# Print out the service status
systemctl --user status interactive_dashboard.service

#You can find error messages in `/home/tajak/Benchmarking-Text-Classification-on-South-Slavic/Interactive-Dashboard` and log messages in `/home/tajak/Benchmarking-Text-Classification-on-South-Slavic/Interactive-Dashboard`