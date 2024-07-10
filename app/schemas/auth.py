from pydantic import BaseModel, Field


class LoginData(BaseModel):
    email: str = Field(description="Your email", examples=["john@smith.com"])
    password: str = Field(description="Your password", examples=["qwerty123!"])


class UserData(LoginData):
    first_name: str = Field(description="Your first name", examples=["John"])
    last_name: str = Field(description="Your last name", examples=["Smith"])


class Token(BaseModel):
    access_token: str
    token_type: str
