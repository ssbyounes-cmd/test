import json
from pydantic import BaseModel


class types(BaseModel):
    type: str


class Prompt(BaseModel):
    prompt: str


class function(BaseModel):
    name: str
    description: str
    parameters: dict[str, types]
    returns: types

def get_prompts():
    with open("data/input/function_calling_tests.json") as f:
        results: list[dict] = json.load(f)
        return [Prompt(**r) for r in results]


def get_functions():
    with open("data/input/functions_definition.json") as f:
        results: list[dict] = json.load(f)
        return [function(**r) for r in results]

    