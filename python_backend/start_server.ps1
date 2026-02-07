$ScriptDir = Split-Path $MyInvocation.MyCommand.Path
Set-Location $ScriptDir
# Local Whisper config (auto-download ~1GB 'medium' model)
$env:LOCAL_WHISPER_MODEL = "medium"
$env:LOCAL_WHISPER_DEVICE = "cpu"
$env:LOCAL_WHISPER_COMPUTE = "int8"
$env:WHISPER_LANG = "en"

# Optional: OpenAI Whisper fallback (uncomment and set your key)
# $env:OPENAI_API_KEY = "YOUR_KEY"
& "C:\Users\Logesh\AppData\Local\Programs\Python\Python312\python.exe" "main.py"
