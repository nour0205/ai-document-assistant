import json
import requests

API_URL = "http://127.0.0.1:8000/ask"
REFUSAL_PHRASE = "I don't know based on the document"


def run():
    with open("evaluation/questions.json", "r") as f:
        questions = json.load(f)

    passed = 0
    failed = 0

    print("\n🧪 Running RAG Evaluation\n")

    for item in questions:
        qid = item["id"]
        question = item["question"]
        expected = item["expected_behavior"]

        response = requests.post(API_URL, json={"question": question})
        answer = response.json()["answer"]

        if expected == "ANSWER":
            success = REFUSAL_PHRASE not in answer
        elif expected == "REFUSE":
            success = REFUSAL_PHRASE in answer
        else:
            success = False

        status = "✅ PASS" if success else "❌ FAIL"

        print(f"{status} | {qid}")
        print(f"Q: {question}")
        print(f"A: {answer}\n")

        if success:
            passed += 1
        else:
            failed += 1

    print("📊 Evaluation Summary")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Total: {passed + failed}")


if __name__ == "__main__":
    run()
