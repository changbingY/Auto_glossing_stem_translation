import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
from typing import List

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class BERTBasedEncoder(nn.Module):
    def __init__(self, projection_dim: int = 512):
        super(BERTBasedEncoder, self).__init__()
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert = BertModel.from_pretrained('bert-base-uncased').to(device)
        #self.reduce_dim = nn.Linear(768, projection_dim).to(device)
    
    def forward(self, inputs: List[str], lengths: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            encoded_input = self.tokenizer(inputs, padding=True, return_tensors="pt").to(device)
            outputs = self.bert(**encoded_input)
            last_hidden_states = outputs.last_hidden_state  # Extract the last hidden state
        #last_hidden_states = self.reduce_dim(last_hidden_states).to(device)
        return last_hidden_states.to(device)  # You may need to do further processing based on the specific requirements of your model



######LLama
# import torch
# import torch.nn as nn
# #from transformers import BertTokenizer, BertModel
# #from transformers import LlamaForCausalLM, LlamaTokenizer
# from typing import List
# from transformers import LlamaTokenizerFast
# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
# from transformers import AutoTokenizer, AutoModel

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# class BERTBasedEncoder(nn.Module):
#     def __init__(self, projection_dim: int = 512):
#         super(BERTBasedEncoder, self).__init__()
#         self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf",token='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')
#         self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})

#         self.bert = AutoModel.from_pretrained("meta-llama/Llama-2-7b-hf",token='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')
#         self.bert.resize_token_embeddings(len(self.tokenizer))
#         #tokenizer = LlamaTokenizerFast.from_pretrained("hf-internal-testing/llama-tokenizer")
#         #self.tokenizer = LlamaTokenizer.from_pretrained("/home/ycblena/scratch/Hard/sigmorphon-2023-glossing_bert_decoder+char_Tsez_modify_llama2/llama-2-7b")
#         #self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
#         #self.bert = LlamaForCausalLM.from_pretrained("/home/ycblena/scratch/Hard/sigmorphon-2023-glossing_bert_decoder+char_Tsez_modify_llama2/llama-2-7b").to(device)
#         #self.bert = BertModel.from_pretrained("/home/ycblena/scratch/Hard/sigmorphon-2023-glossing_bert_decoder+char_Tsez_modify_llama2/llama-2-7b/consolidated.00.pth").to(device)
#         #self.reduce_dim = nn.Linear(4096, projection_dim).to(device)
    
#     def forward(self, inputs: List[str], lengths: torch.Tensor) -> torch.Tensor:
#         with torch.no_grad():
#             encoded_input = self.tokenizer(inputs, padding=True, return_tensors="pt").to(device)
#             outputs = self.bert(**encoded_input)
#             last_hidden_states = outputs.last_hidden_state

#         #reduced_states = self.reduce_dim(last_hidden_states).to(device)
#         return last_hidden_states.to(device)

#             #last_hidden_states = outputs.last_hidden_state  # Extract the last hidden state
#         #last_hidden_states = self.reduce_dim(last_hidden_states).to(device)
#         #return last_hidden_states.to(device)  # You may need to do further processing based on the specific requirements of your model
