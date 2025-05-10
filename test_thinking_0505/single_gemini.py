from google import genai
from google.genai import types

#my_api_key_p1 = "AIzaSyC4SgiM-"
#my_api_key_p2 = "OXAtW0UjbepH3GBLHV-ShNOF_E"
#my_api_key = my_api_key_p1 + my_api_key_p2
# my_api_key = "AIzaSyAUH70gKFSmR52QAbZq4fJFM3WSbTYCHp8"
# my_api_key = "AIzaSyAI4NUbl2swWAqSmqWZ0f-8KvaP-Oo_IYA"
my_api_key = "AIzaSyDTST5wSG3mw0kbCDy3i3krUVyBTaFTRfQ"
client = genai.Client(api_key=my_api_key)

response = client.models.generate_content(
    model="gemini-2.5-pro-exp-03-25",
    contents="Find the number of ordered pairs $(x,y)$, where both $x$ and $y$ are integers "
             "between $-100$ and $100$ inclusive, such that $12x^2-xy-6y^2=0$.",
    config=types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(thinking_budget=8000)
    ),
)

print(response.text)

