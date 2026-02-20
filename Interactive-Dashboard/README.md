# Interactive Dashboard

The dashboard is available at https://www.clarin.si/classla-llm-dashboard/

## Instructions for dashboard manager

### Running the dashboard through a service manager (for the CLARIN.SI website)

Option 1 (recommended): `bash /home/tajak/Benchmarking-Text-Classification-on-South-Slavic/Interactive-Dashboard/re-run-service.sh` (a script that contains all the steps listed below)

Option 2 (Step-by-step): run the following commands (from the initial working directory (`\`)):
- assure that no service is running, disable any running service: `systemctl --user disable interactive_dashboard.service` and reload: `systemctl --user daemon-reload`
- enable the service: `systemctl --user enable interactive_dashboard.service`
- start the service: `systemctl --user start interactive_dashboard.service`
- to check the status:
	- `systemctl --user status interactive_dashboard.service` 
	- or `journalctl --user -xeu interactive_dashboard.service`
- to stop the service: `systemctl --user stop interactive_dashboard`

You can find error messages in `Interactive-Dashboard/app.error` and log messages in `Interactive-Dashboard/app.log`

### Running the dashboard locally

Use the `interactive_dash` conda environment.
Or, create a new conda environment and install `pip install streamlit pandas plotly`

To run the dashboard:
- move to the Interactive-Dashboard directory: `Benchmarking-Text-Classification-on-South-Slavic/Interactive-Dashboard`
- first, create a screen: `screen -S classla-dashboard`
- then activate the conda environment: `conda activate interactive_dash`
- then run the streamlit app: `streamlit run app.py > app.log`
- detach from the screen: CTRL + A + D

To re-attach to the screen: `screen -r classla-dashboard` (to close the screen: CTRL + A + D)

Server information:
- port = 48883
- address = "0.0.0.0"