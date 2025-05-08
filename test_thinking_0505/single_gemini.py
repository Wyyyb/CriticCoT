from google import genai
from google.genai import types

my_api_key_p1 = "AIzaSyC4SgiM-"
my_api_key_p2 = "OXAtW0UjbepH3GBLHV-ShNOF_E"
my_api_key = my_api_key_p1 + my_api_key_p2
client = genai.Client(api_key=my_api_key)

response = client.models.generate_content(
    model="gemini-2.5-pro-exp-03-25",
    contents="Find the number of ordered pairs $(x,y)$, where both $x$ and $y$ are integers "
             "between $-100$ and $100$ inclusive, such that $12x^2-xy-6y^2=0$.",
    config=types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(thinking_budget=16384)
    ),
)

print(response.text)

