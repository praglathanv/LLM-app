from dotenv import load_dotenv
load_dotenv()

from groq import Groq
import json

client = Groq()

# --- Fake DB ---
candidates_db = {
    "101": {"name": "John Doe", "start_year": 2019, "skills": ["Python", "SQL"]},
    "102": {"name": "Priya Sharma", "start_year": 2021, "skills": ["React", "Node.js"]}
}

# --- Tools ---
def get_candidate_by_id(candidate_id: str) -> str:
    c = candidates_db.get(candidate_id)
    return json.dumps(c) if c else "Candidate not found"

def calculate_experience_years(start_year: int) -> str:
    return f"{2026 - start_year} years"

def extract_candidate(name: str, years_experience: int, skills: list) -> str:
    new_id = str(100 + len(candidates_db) + 1)
    candidates_db[new_id] = {
        "name": name,
        "start_year": 2026 - years_experience,
        "skills": skills
    }
    return f"Added candidate {name} with ID {new_id}"

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_candidate_by_id",
            "description": "Look up a candidate's details by their ID",
            "parameters": {
                "type": "object",
                "properties": {"candidate_id": {"type": "string"}},
                "required": ["candidate_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_experience_years",
            "description": "Calculate years of experience from a start year",
            "parameters": {
                "type": "object",
                "properties": {"start_year": {"type": "integer"}},
                "required": ["start_year"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "extract_candidate",
            "description": "Extract structured candidate info from free text and save it to the database",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "years_experience": {"type": "integer"},
                    "skills": {"type": "array", "items": {"type": "string"}}
                },
                "required": ["name", "years_experience", "skills"]
            }
        }
    }
]

available_functions = {
    "get_candidate_by_id": get_candidate_by_id,
    "calculate_experience_years": calculate_experience_years,
    "extract_candidate": extract_candidate
}

messages = [{"role": "system", "content": "You are a recruiting assistant. Use tools when needed."}]

print("Recruiting Assistant (type 'exit' to quit)")

while True:
    user_input = input("\nYou: ")
    if user_input.lower() == "exit":
        break

    messages.append({"role": "user", "content": user_input})

    while True:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            tools=tools
        )
        msg = response.choices[0].message
        messages.append(msg)

        if not msg.tool_calls:
            print("Assistant:", msg.content)
            break

        for tool_call in msg.tool_calls:
            func_name = tool_call.function.name
            args = json.loads(tool_call.function.arguments)
            result = available_functions[func_name](**args)
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result
            })