import json
import os


def main(input_dir, output_path):
    input_data = []
    output_data = []
    for file in os.listdir(input_dir):
        if not file.endswith(".json"):
            continue
        file_path = os.path.join(input_dir, file)
        with open(file_path, "r") as f:
            data = json.load(f)
        output_data
    print("len(input_data)", len(input_data))

    for each in input_data:
        critique = each.get("critique", None)
        if not critique:
            continue
        # if len(critique) > 20000 and "</think>" not in critique:
            # print("len(critique)", len(critique))
            # print(critique)
            # continue
        output_data.append(each)
    print("len(output_data)", len(output_data))
    with open(output_path, "w") as f:
        f.write(json.dumps(output_data, indent=4))


if __name__ == "__main__":
    my_input_dir = "seed_questions/"
    my_output_path = "sft_merged_to_call_api.json"
    main(my_input_dir, my_output_path)

