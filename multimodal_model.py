import torch
import torch.nn as nn
import torch.nn.functional as F
from co_dual_cross_attention import CoDualCrossAttentionModel


class SelfAttention(nn.Module):
    def __init__(self, input_dim=512, output_dim=256):
        super(SelfAttention, self).__init__()

        self.query = nn.Linear(input_dim, output_dim)
        self.key = nn.Linear(input_dim, output_dim)
        self.value = nn.Linear(input_dim, output_dim)

    def forward(self, patches):
        q = self.query(patches)
        k = self.key(patches)
        v = self.value(patches)

        attention_scores = torch.matmul(q, k.transpose(-2, -1))
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_output = torch.matmul(attention_probs, v)
        

        # Summing over the patches to get a single representation
        #return attention_output.max(dim=1)[0]

        return attention_output.sum(dim=1)

class MultimodalModel(nn.Module):
    def __init__(self, feat_out=2048, output_dim=128):
        super(MultimodalModel, self).__init__()
        
        self.feat_out = feat_out
        self.output_dim = output_dim
        
        self.self_attention = SelfAttention(self.feat_out, self.feat_out)
        self.comodel = CoDualCrossAttentionModel(img_dim=256, gene_dim=138, co_output_dim=256, cross_output_dim=256, nhead=8, num_layers=3, dim_feedforward=512)
        
    
        comodel_output_dim = 512

        self.fc1 = nn.Sequential(
            nn.Linear(comodel_output_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.1) 
        )
        
        self.fc2 = nn.Sequential(
            nn.Linear(512 , 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.1) 
        )
        
        
        self.fc3 = nn.Sequential(
            nn.Linear(256,128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.01) 
        )
        # After adding clinical data 
        self.fc4 = nn.Sequential(
            nn.Linear(128+4, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Dropout(0.01) 
        )
        
        self.final_layer = nn.Sequential(
            nn.Linear(32, 1, bias=False),  
        )


    def forward(self, image_patches, gene_expression, clinical_data):
        # Assuming clinical_data is a tensor of shape [batch_size, 6]
        #print(image_patches.shape)
        image_embedding = self.self_attention(image_patches)

        features =  self.comodel(image_embedding, gene_expression)
      
        x = self.fc1(features)
        x = self.fc2(x)
        x = self.fc3(x)

        x = torch.cat((x, clinical_data), dim=1) # Added 4 for clinical features
        x = self.fc4(x)

        output = self.final_layer(x)
        return output
