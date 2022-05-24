import os
import json
import argparse

success = 0
queue_empty = 0
query_limit = 0
step_timeout_limit = 0
illegal_proof = 0
timeout = 0
total = 0

parser = argparse.ArgumentParser(description='Analyse evaluation results.')
parser.add_argument('--result-dir', '-rd', type=str)
args = parser.parse_args()

if os.path.isdir("temp_results"):
    os.system("rm temp_results/*")
else:
    os.system("mkdir temp_results")
download_command = f"gsutil -m cp {args.result_dir}/*.json temp_results/"
delete_command = "rm temp_results/*.json"
os.system(download_command)

tested_indices = []
for file in os.listdir("temp_results"):
    if file.endswith("json"):
        index = int(file.split(".")[0])
        tested_indices.append(index)

        content = json.load(open(os.path.join("temp_results", file)))
        if "result" not in content["agent_info"]:
            continue
        total += 1
        if content["proved"] and "\noops" not in content["proof_script"] and "sorry" not in content["proof_script"]:
            success += 1
            print(content["proof_script"])
            print("#" * 150)
            continue
        if content["agent_info"]["result"] == "query limit reached":
            query_limit += 1
            continue
        if content["agent_info"]["result"] == "timeout":
            timeout += 1
            continue
        if content["agent_info"]["result"] == "queue empty":
            queue_empty += 1
            continue
        if content["agent_info"]["result"] == "step timeout limit reached":
            step_timeout_limit += 1
            continue
        if "\noops" in content["proof_script"] or "sorry" not in content["proof_script"]:
            illegal_proof += 1

assert total == success + query_limit + timeout + queue_empty + step_timeout_limit + illegal_proof
print(f"Total: {total}, success: {success/total}, query limit reached: {query_limit/total}, timeout: {timeout/total}, "
      f"queue empty: {queue_empty/total}, step timeout limit: {step_timeout_limit/total}, illegal proof: {illegal_proof/total}.")
os.system(delete_command)


untested_indices = []
max_index = max(tested_indices)
tested_indices = set(tested_indices)
for i in range(max_index + 1):
    if i not in tested_indices:
        untested_indices.append(str(i))

print("The following indexed problems are not tested:\n" + ", ".join(untested_indices))

