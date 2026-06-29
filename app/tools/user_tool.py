import requests
import os
import json

user = requests.get("https://jsonplaceholder.typicode.com/todos")

try:
    
    make_dir = os.makedirs("user-data", exist_ok=True)
    
    if make_dir:
        with open("user-data/user.txt", "w") as f:
            f.write(user.text)
        print("Successfully created user-data/user.txt")
    
    response = user.json()

    with open("user-data/user.json", "w") as f:
        json.dump(response, f, indent=4)
    print("Successfully created user-data/user.json")

except ValueError:
    print("Invalid JSON response")

except Exception as e:
    print(f"Error fetching data: {e}")

