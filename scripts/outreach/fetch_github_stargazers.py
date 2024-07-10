import csv
import os

import requests

# Replace 'your_github_token_here' with your actual GitHub token
GITHUB_TOKEN = ""
REPO_OWNER = "IBM"
REPO_NAME = "terratorch"
CSV_FILE = "terratorch_stargazers_info.csv"


def get_stargazers(repo_owner, repo_name):
    url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/stargazers"
    headers = {"Authorization": f"token {GITHUB_TOKEN}", "Accept": "application/vnd.github.v3+json"}
    stargazers = []
    page = 1

    while True:
        response = requests.get(url, headers=headers, params={"page": page, "per_page": 100})
        if response.status_code != 200:
            break
        data = response.json()
        if not data:
            break
        stargazers.extend(data)
        page += 1

    return stargazers


def get_user_info(username):
    url = f"https://api.github.com/users/{username}"
    headers = {"Authorization": f"token {GITHUB_TOKEN}", "Accept": "application/vnd.github.v3+json"}
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        return response.json()
    return None


def collect_stargazer_info(stargazers):
    collected_info = []

    for user in stargazers:
        user_info = get_user_info(user["login"])
        if user_info:
            collected_info.append(
                {
                    "person_name": user_info.get("name", ""),
                    "github_profile_link": user_info.get("html_url", ""),
                    "email": user_info.get("email", ""),
                    "company": user_info.get("company", ""),
                    "interests": user_info.get("bio", ""),  # Assuming interests might be mentioned in the bio
                }
            )

    return collected_info


def save_to_csv(data, filename):
    keys = data[0].keys()
    with open(filename, "w", newline="", encoding="utf-8") as output_file:
        dict_writer = csv.DictWriter(output_file, fieldnames=keys)
        dict_writer.writeheader()
        dict_writer.writerows(data)


def main():
    stargazers = get_stargazers(REPO_OWNER, REPO_NAME)
    if stargazers:
        stargazer_info = collect_stargazer_info(stargazers)
        save_to_csv(stargazer_info, CSV_FILE)
        print(f"Data saved to {CSV_FILE}")
    else:
        print("No stargazers found or failed to fetch data.")


if __name__ == "__main__":
    main()
