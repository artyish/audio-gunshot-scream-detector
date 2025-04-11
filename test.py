import gdown

file_id = "1GYWZ3sgidwDOSOd2xIBXqKeIIWbJjsWj"
url = f"https://drive.google.com/uc?id={file_id}"
output = "mainmodel.keras"

gdown.download(url, output, quiet=False)