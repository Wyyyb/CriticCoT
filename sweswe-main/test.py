import importlib

chat_scheduler = "examples.ppo_trainer.r2e_chat_scheduler.R2EChatCompletionScheduler"
module_path, class_name = chat_scheduler.rsplit(".", 1)
print(module_path, class_name)
module = importlib.import_module(module_path)
print(module)
scheduler_cls = getattr(module, class_name)