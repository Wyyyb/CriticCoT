# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Evaluation functions for BigBench Extra Hard."""


def strip_latex(response: str) -> str:
  if response.startswith("$") and response.endswith("$"):
    response = response[1:-1]
  if "boxed{" in response and response.endswith("}"):
    response = response[0:-1].split("boxed{")[1]
  if "text{" in response and response.endswith("}"):
    response = response[0:-1].split("text{")[1]
  if "texttt{" in response and response.endswith("}"):
    response = response[0:-1].split("texttt{")[1]
  return response


def extract_answer(sample: str) -> str:
  """Extracts the final answer from the sample."""
  answer_prefixes = [
      "The answer is:",
      "The final answer is ",
      "The final answer is: ",
      "The answer is ",
      "Final answer: "
  ]
  answer = sample
  for answer_prefix in answer_prefixes:
    if answer_prefix in answer:
      answer = answer.split(answer_prefix)[-1].strip()
  if answer.endswith("."):
    answer = answer[:-1]
  return strip_latex(answer)


def fuzzy_match(prediction: str, reference: str) -> bool:
  """Fuzzy match function for BigBench Extra Hard."""
  if prediction == reference:
    return True

  # (a) vs a
  if len(prediction) == 3 and prediction[0] == "(" and prediction[-1] == ")":
    return prediction[1] == reference
  if len(reference) == 3 and reference[0] == "(" and reference[-1] == ")":
    return reference[1] == prediction

  # Numbers
  try:
    if float(prediction) == float(reference):
      return True
  except ValueError:
    pass

  # quote issues
  if prediction.replace("'", "") == reference.replace("'", ""):
    return True

  # Bracket issues
  if f"[{reference}]" == prediction or f"[{prediction}]" == reference:
    return True

  # Question mark issues
  if prediction.endswith("?") and prediction[:-1] == reference:
    return True

  return False


def preprocess_sample(sample: str) -> str:
  prediction = extract_answer(sample.strip()).lower()
  prediction = prediction.replace(", ", ",").replace("**", "")
  prediction = prediction.split("\n")[0]
  prediction = prediction[0:-1] if prediction.endswith(".") else prediction
  return prediction


def preprocess_reference(reference: str) -> str:
  reference = reference.strip().lower()
  reference = reference.replace(", ", ",")
  return reference


def evaluate_correctness(sample: str, reference: str) -> bool:
  prediction = preprocess_sample(sample)
  reference = preprocess_reference(reference)
  return fuzzy_match(prediction, reference)


# Examples
# print(evaluate_correctness("Ok The final answer is: \\boxed{4}.", "4"))
# print(evaluate_correctness("[Reasoning] The final answer is: \\boxed{4}.", "3"))
# print(evaluate_correctness("Alright! The final answer is: 2, 3, 4", "2,3,4"))
# print(evaluate_correctness("blah blah The final answer is: 2, 3, 4", "2,3,5"))
# print(evaluate_correctness("Ok The answer is: \\boxed{(A)}", "a"))
# print(evaluate_correctness("Ok The answer is: (A)", "b"))
# print(evaluate_correctness("Ok The answer is: **25**\nHere's why.", "25.0"))
# print(evaluate_correctness("Ok The answer is: **25**\nHere's why.", "26.0"))
