from typing import Any, Dict, List, Optional

from ..math.math_handler import MathTaskHandler


class AIMETaskHandler(MathTaskHandler):
    def generate_prompt(self, problem: Dict):
        return self.task_config.templating_parameters["template"].format(
            prompt=problem["problem"]
        )

    def make_conversations(
        self,
        data: List[Dict[str, Any]],
        system_prompt: Optional[str] = None,
        user_template: Optional[str] = None,
    ):
        conversations = []
        for problem in data:
            prompt_text = self.generate_prompt(problem)
            conversations.append(
                self.make_conversation_from_contents(
                    [prompt_text],
                    system_prompt=system_prompt,
                    user_template=user_template,
                )
            )
        return conversations

    def load_and_filter_dataset(
        self, start, end, split=None, subset=None, difficulty=None, args=None
    ):
        train_data = self.load_dataset(subset=subset, split=split).to_pandas()
        filtered_data = train_data[train_data["url"].str.contains("2024", na=False)]
        return filtered_data.iloc[start:end] if end > 0 else filtered_data.iloc[start:]
