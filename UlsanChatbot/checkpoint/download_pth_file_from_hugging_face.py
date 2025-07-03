import requests

files = {
    "deglok-text-classification.pth": "https://huggingface.co/mindsignal/uwins_chabot_project/resolve/main/deglok-text-classification.pth",
    "hubok-text-classification.pth": "https://huggingface.co/mindsignal/uwins_chabot_project/resolve/main/hubok-text-classification.pth",
}

for filename, url in files.items():
    response = requests.get(url)
    response.raise_for_status()
    with open(filename, "wb") as f:
        f.write(response.content)
    print(f"{filename} downloaded successfully.\n")