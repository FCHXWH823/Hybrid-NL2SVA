from vllm.entrypoints.llm import LLM
from vllm.sampling_params import SamplingParams

llm = LLM(model="tiiuae/falcon-7b-instruct")

sampling_params = SamplingParams(temperature = 0.9, 
                                 max_tokens = 200)

prompt = "What is quantum computing?"
output = llm.generate(prompt, sampling_params)

print(output)
print(output[0].outputs[0].text)