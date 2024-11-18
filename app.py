from flask import Flask, request, jsonify
import warnings
import json
import re
from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer, GenerationConfig, pipeline
from langchain import HuggingFacePipeline
from langchain_core.output_parsers import StrOutputParser

warnings.filterwarnings('ignore')

# Flask app initialization
app = Flask(__name__)

# Load the model and tokenizer
MODEL_NAME = "model"

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype="float16",
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, torch_dtype="float16",
    trust_remote_code=True,
    device_map="auto",
    quantization_config=quantization_config
)

generation_config = GenerationConfig.from_pretrained(MODEL_NAME)
generation_config.max_new_tokens = 1024
generation_config.temperature = 0.1
generation_config.top_p = 0.95
generation_config.do_sample = True
generation_config.repetition_penalty = 1.15

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    return_full_text=True,
    generation_config=generation_config,
)

llm = HuggingFacePipeline(pipeline=pipe)
output_parser = StrOutputParser()

# Flask route to handle inference
@app.route("/codestral", methods=["POST"])
def codestral():
    try:
        # Parse input JSON
        data = request.get_json()
        codes = data.get("codes", [])
        prompt = data.get("prompt", "")

        # Convert codes to JSON string
        codes_object = json.dumps(codes)

        # Create prompt for the model
        to_model = f"Code list: \n {codes_object} {prompt}"
        output = llm | output_parser
        response = output.invoke(to_model)

        # Extract JSON from response
        json_pattern = r"\[\s*{.*?}\s*\]"
        matches = re.findall(json_pattern, response, re.DOTALL)

        json_objects = []
        for json_text in matches:
            try:
                json_data = json.loads(json_text)
                json_objects.append(json_data)
            except json.JSONDecodeError:
                print("Error: Skipping invalid JSON object.")

        return jsonify({"response": json_objects})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Run the Flask app
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
