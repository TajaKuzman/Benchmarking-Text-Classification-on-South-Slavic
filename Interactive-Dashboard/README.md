# Interactive Dashboard

## Run the dashboard

Use the `interactive_dash` conda environment.
Or, create a new conda environment and install `pip install streamlit pandas plotly`

To run the dashboard:
- move to the Interactive-Dashboard directory: `/home/tajak/Benchmarking-Text-Classification-on-South-Slavic/Interactive-Dashboard`
- first, create a screen: `screen -S classla-dashboard`
- then activate the conda environment: `conda activate interactive_dash`
- then run the streamlit app: `streamlit run app.py > app.log`
- detach from the screen: CTRL + A + D

To re-attach to the screen: `screen -r classla-dashboard` (to close the screen: CTRL + A + D)

Server information:
- port = 48883
- address = "0.0.0.0"