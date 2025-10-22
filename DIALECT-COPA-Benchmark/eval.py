import os
import pandas as pd
import json
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
import argparse

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("submission", help="Absolute path to the folder with submission files in JSON format")
	args = parser.parse_args()

submission_folder = args.submission

def testing_accuracy(true, pred):
    """
    This function takes the list of true labels and list of predictions and evaluates the model based on comparing them.
    It calculates accuracy.
    
    Args:
    - y_true: list of true labels
    - y_pred: list of predicted labels

    The function returns a dictionary with accuracy.
    """
    y_true = true
    y_pred = pred

    # Calculate the scores
    accuracy = accuracy_score(y_true, y_pred)
    
    return {"Accuracy": accuracy}

#Calculate the scores
def testing(true, pred, labels):
    """
    This function takes the list of true labels and list of predictions and evaluates the model based on comparing them.
    It calculates micro and macro F1 scores.
    
    Args:
    - y_true: list of true labels
    - y_pred: list of predicted labels

    The function returns a dictionary with micro and macro F1.
    """
    y_true = true
    y_pred = pred
    LABELS = labels

    # Calculate the scores
    macro = f1_score(y_true, y_pred, labels=LABELS, average="macro")
    micro = f1_score(y_true, y_pred, labels=LABELS,  average="micro")
    accuracy = accuracy_score(y_true, y_pred)
    #print(f"Macro f1: {macro:0.3}, Micro f1: {micro:0.3}")
    
    return {"Accuracy": accuracy, "Micro F1": micro, "Macro F1": macro}

# Open the jsonl file with all results
with open("results/results.json", "r") as result_file:
    results_list = json.load(result_file)

# Get paths to all the submission files
submission_files = os.listdir(submission_folder)

# Evaluate all submissions in the submissions directory
for submission_file in submission_files:
    # Use only files that start with "submission"
    if "submission-" in submission_file:
        # Open the submission to be evaluated
        with open("{}/{}".format(submission_folder,submission_file), "r") as sub_file:
            results = json.load(sub_file)

            # Get information on the dataset and the model
            model = results["system"]

            dataset_name = results["predictions"][0]["test"]

            # Calculate overall results
            y_true_file = open("datasets/test_labels.txt", "r").readlines()
            y_true = [int(x.replace("\n", "")) for x in y_true_file]

            y_pred = results["predictions"][0]["predictions"]

            current_scores = testing_accuracy(y_true, y_pred)

            # Calculate results for each language
            language_results_dict = {}
            eval_lang = dataset_name.replace("copa-", "")

            language_results_dict[eval_lang] = {"Accuracy": float(current_scores["Accuracy"])}

            current_res_dict = {"Model": model, "Test Dataset": dataset_name, "Accuracy": current_scores["Accuracy"], "Language-Specific Scores": language_results_dict}

            # Add the results to all results
            results_list.append(current_res_dict)

            with open("results/results.json", "w") as new_result_file:
                json.dump(results_list, new_result_file, indent = 2)

    else:
        print("Error: the following file `{}` is either not a submission file or is incorrectly named - see the `README.md` on how to prepare submission files.")

print("All evaluations completed. The results are added to the `results/results.json` file.")


# Create a dataframe from all results

result_df = pd.DataFrame(results_list)

# For each dataset, create a table with results

def results_table(result_df, dataset):
    dataset_df = result_df[result_df["Test Dataset"] == dataset]

    # Sort values based on highest Macro F1
    dataset_df = dataset_df.sort_values(by="Accuracy", ascending=False)

    # Round scores to 3 decimal places
    dataset_df["Accuracy"] = dataset_df["Accuracy"].round(3)
    dataset_df.drop(columns=["Language-Specific Scores"], inplace=True)

    print(dataset_df.to_markdown(index=False))

    return dataset_df


for dataset in ["copa-en", "copa-sl", "copa-hr", "copa-hr-ckm", "copa-mk", "copa-sl-cer", "copa-sr", "copa-sr-tor"]:
    print("New benchmark scores:\n")

    current_df = results_table(result_df, dataset)
    
    print("\n------------------------------------------\n")

    # Save the table in markdown
    with open("results/results-{}.md".format(dataset), "w") as result_file:
        result_file.write("## {}\n\n".format(dataset))
        result_file.write(current_df.to_markdown(index=False))

# Create language-specific results
lang_results_dict = []

for lang in ["en", "sl", "hr", "hr-ckm", "mk", "sl-cer", "sr", "sr-tor"]:
	for result in results_list:
		cur_result = {"Model": result["Model"], "Test Dataset": result["Test Dataset"], "Language": lang}
		try:
			cur_accuracy = result["Language-Specific Scores"][lang]["Accuracy"]
			cur_result["Accuracy"] = cur_accuracy

			lang_results_dict.append(cur_result)
		except:
			continue

lang_results_df = pd.DataFrame(lang_results_dict)

# For each language, create a table with results

def results_table_lang(lang_results_df, lang):
    dataset_df = lang_results_df[lang_results_df["Language"] == lang]

    # Sort values based on highest Accuravy
    dataset_df = dataset_df.sort_values(by="Accuracy", ascending=False)

    # Round scores to 3 decimal places
    dataset_df["Accuracy"] = dataset_df["Accuracy"].round(3)
    

    print(dataset_df.to_markdown(index=False))

    return dataset_df


lang_result_file = open("results/language-specific-results.md", "w")

for lang in ["en", "sl", "hr", "hr-ckm", "mk", "sl-cer", "sr", "sr-tor"]:

    current_df = results_table_lang(lang_results_df, lang)
    
    lang_result_file.write("\n#### {}\n\n".format(lang))
    lang_result_file.write(current_df.to_markdown(index=False))
    lang_result_file.write("\n\n------------------------------------------\n")

lang_result_file.close()