import csv
import os

import requests

# Replace 'your_github_token_here' with your actual GitHub token
GITHUB_TOKEN = ""
REPO_OWNER = "IBM"
REPO_NAME = "terratorch"
CSV_FILE = "terratorch_forkers_info.csv"


def get_forks(repo_owner, repo_name):
    url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/forks"
    headers = {"Authorization": f"token {GITHUB_TOKEN}", "Accept": "application/vnd.github.v3+json"}
    forkers = []
    page = 1

    while True:
        response = requests.get(url, headers=headers, params={"page": page, "per_page": 100})
        if response.status_code != 200:
            break
        data = response.json()
        if not data:
            break
        forkers.extend(data)
        page += 1

    return forkers


def get_user_info(username):
    url = f"https://api.github.com/users/{username}"
    headers = {"Authorization": f"token {GITHUB_TOKEN}", "Accept": "application/vnd.github.v3+json"}
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        return response.json()
    return None


def collect_forker_info(forkers):
    collected_info = []

    for fork in forkers:
        user_info = get_user_info(fork["owner"]["login"])
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
    if not data:
        print("No data to save.")
        return

    keys = data[0].keys()
    with open(filename, "w", newline="", encoding="utf-8") as output_file:
        dict_writer = csv.DictWriter(output_file, fieldnames=keys)
        dict_writer.writeheader()
        dict_writer.writerows(data)


def main():
    forkers = get_forks(REPO_OWNER, REPO_NAME)
    if forkers:
        forker_info = collect_forker_info(forkers)
        save_to_csv(forker_info, CSV_FILE)
        print(f"Data saved to {CSV_FILE}")
    else:
        print("No forkers found or failed to fetch data.")


if __name__ == "__main__":
    main()
