import os
from modules.generator.question_answering import QAGeneratorMistral

# Test with simple, short context
API_KEY = os.environ["MISTRAL_API_KEY"]
qa_generator = QAGeneratorMistral(api_key=API_KEY)

simple_context = ["Green tea contains antioxidants that help prevent disease."]
simple_query = "What do antioxidants do?"

try:
    print("Testing QAGeneratorMistral directly...")
    answer = qa_generator.generate_answer(simple_query, simple_context)
    print(f"Success! Answer: {answer}")
except Exception as e:
    print(f"Direct test failed: {e}")
    print(f"Exception type: {type(e).__name__}")