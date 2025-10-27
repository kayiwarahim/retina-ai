import requests

# URL of your running Flask app
url = "http://127.0.0.1:10000/predict"

# Pick any image you want to test
files = {'file': open('gla.jpg', 'rb')}

response = requests.post(url, files=files)
print(response.json())
