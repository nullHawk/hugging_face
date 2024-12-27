from transformers import AutoModel
from transformers import AutoTokenizer

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
model = AutoModel.from_pretrained(checkpoint)

tokenizer = AutoTokenizer.from_pretrained(checkpoint)
inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
outputs = model(**inputs)
print(outputs.last_hidden_state.shape)
print(outputs)