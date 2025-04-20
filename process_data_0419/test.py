
def get_prompt(question):
    question = "Question:\n" + question
    prompt = f"<|im_start|>Please reason step by step to find a solution to the following question, " \
             f"and put your final answer within \\boxed{{}}.<|im_end|>\n\n" \
             f"<|im_start|>user\n{question}<|im_end|>\n" \
             f"<|im_start|>assistant\n"
    return prompt


def main():
    question = "QUESTION"
    p = get_prompt(question)
    print("p", p)


main()

