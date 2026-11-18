
import requests

# Send a GET request to the root endpoint.
r = requests.get("http://127.0.0.1:8000/", timeout=30)

# Print the GET status code and welcome message.
print("GET Request")
print(f"Status Code: {r.status_code}")
print(f"Welcome Message: {r.json()['message']}")


# Sample census data for the prediction request.
data = {
    "age": 37,
    "workclass": "Private",
    "fnlgt": 178356,
    "education": "HS-grad",
    "education-num": 10,
    "marital-status": "Married-civ-spouse",
    "occupation": "Prof-specialty",
    "relationship": "Husband",
    "race": "White",
    "sex": "Male",
    "capital-gain": 0,
    "capital-loss": 0,
    "hours-per-week": 40,
    "native-country": "United-States",
}

# Send a POST request containing the sample census data.
r = requests.post(
    "http://127.0.0.1:8000/data/",
    json=data,
    timeout=30,
)

# Print the POST status code and prediction result.
print("\nPOST Request")
print(f"Status Code: {r.status_code}")
print(f"Prediction Result: {r.json()['result']}")
