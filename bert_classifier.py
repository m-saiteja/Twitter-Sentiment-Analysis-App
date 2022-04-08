from torch import nn
from transformers import BertModel

class BertClassifierModel(nn.Module):
    
    def __init__(self, d_hidden = 768, bert_variant = "bert-base-uncased"):
        """
        Define the architecture of Bert-Based classifier.
        You will mainly need to define 3 components, first a BERT layer
        using `BertModel` from transformers library,
        a linear layer to map the representation from Bert to the output,
        and a sigmoid layer to map the score to a proability
        
        Inputs:
            - d_hidden (int): Size of the hidden representations of bert
            - bert_variant (str): BERT variant to use
        """
        super(BertClassifierModel, self).__init__()
        self.bert_layer = BertModel.from_pretrained(bert_variant)
        self.output_layer = nn.Linear(d_hidden, 3)
        self.sigmoid_layer = nn.LogSoftmax()
        

    def forward(self, input_ids, attn_mask):
        """
        Forward Passes the inputs through the network and obtains the prediction
        
        Inputs:
            - input_ids (torch.tensor): A torch tensor of shape [batch_size, seq_len]
                                        representing the sequence of token ids
            - attn_mask (torch.tensor): A torch tensor of shape [batch_size, seq_len]
                                        representing the attention mask such that padded tokens are 0 and rest 1
                                        
        Returns:
          - output (torch.tensor): A torch tensor of shape [batch_size,] obtained after passing the input to the network                                                
        """
        
        output = self.bert_layer(input_ids, attn_mask)
        output = self.output_layer(output.pooler_output)
        output = self.sigmoid_layer(output)
        
        return output.squeeze(-1)
