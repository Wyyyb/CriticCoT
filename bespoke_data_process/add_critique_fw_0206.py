from firework_r1_call_0206 import *
import json


def format_single_query(question, answer):
    ins = "You are a science expert. A student is trying to solve a question, please explain briefly whether his solution is correct or not. Finally, conclude your judgement with 'Conclusion: right/wrong [END]'."
    query = f"{ins}\n\nQuestion:\n{question}\nSolution:\n{answer}"
    return query


def main():
    







