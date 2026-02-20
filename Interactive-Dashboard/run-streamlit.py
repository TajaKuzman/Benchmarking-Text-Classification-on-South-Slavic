from streamlit.web import cli

if __name__ == "__main__":
    # code for running the app on a local path
    #cli.main_run(["/home/tajak/Benchmarking-Text-Classification-on-South-Slavic/Interactive-Dashboard/app.py", "--server.address=172.20.0.170", "--server.port=8502"])
    cli.main_run([
		"/home/tajak/Benchmarking-Text-Classification-on-South-Slavic/Interactive-Dashboard/app.py",
		"--server.address=172.20.0.170",
		#"--server.port=8501",
		"--server.port=8502",
		#"--browser.serverAddress=www.clarin.si",
		#"--browser.serverAddress=beta.clarin.si",
		#"--server.baseUrlPath=classla-llm-dashboard",
		#"--server.headless=true",
		#"--browser.gatherUsageStats=false",
		#"--server.showEmailPrompt=false",
		"--server.enableCORS=false",
		"--server.enableXsrfProtection=false",
		"--server.enableWebsocketCompression=false"
		])