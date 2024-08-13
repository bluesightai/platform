import os

from dotenv import load_dotenv

# Load the .env file
load_dotenv()

# Get the password from the .env file
password = os.getenv("SUPABASE_POSTGRES_PASSWORD")


postgres_uri = (
    f"postgresql://postgres.biccczfztgnfaqzmizan:{password}@aws-0-us-east-1.pooler.supabase.com:6543/postgres"
)
