import torch
import torch.nn as nn
import torch.nn.functional as F


class CoAttention(nn.Module):
    def __init__(self, img_dim, gene_dim, output_dim):
        super(CoAttention, self).__init__()

        # For image features
        self.img_query = nn.Linear(img_dim, output_dim, bias=False)
        self.img_key = nn.Linear(img_dim, output_dim, bias=False)
        self.img_value = nn.Linear(img_dim, output_dim, bias=False)

        # For gene features
        self.gene_query = nn.Linear(gene_dim, output_dim, bias=False)
        self.gene_key = nn.Linear(gene_dim, output_dim, bias=False)
        self.gene_value = nn.Linear(gene_dim, output_dim, bias=False)

    def forward(self, img_features, gene_features):
        # Image attends to Gene
        q_img = self.img_query(img_features)
        k_gene = self.gene_key(gene_features)
        v_gene = self.gene_value(gene_features)

        # Gene attends to Image
        q_gene = self.gene_query(gene_features)
        k_img = self.img_key(img_features)
        v_img = self.img_value(img_features)

        # Compute attention
        attention_img_gene = torch.matmul(F.softmax(torch.matmul(q_img, k_gene.transpose(-2, -1)), dim=-1), v_gene)
        attention_gene_img = torch.matmul(F.softmax(torch.matmul(q_gene, k_img.transpose(-2, -1)), dim=-1), v_img)

        return attention_img_gene, attention_gene_img



class ScaledDotProductAttention(nn.Module):
    def __init__(self, dim, output_dim):
        super(ScaledDotProductAttention, self).__init__()
        self.query = nn.Linear(dim, output_dim)
        self.key = nn.Linear(dim, output_dim)
        self.value = nn.Linear(dim, output_dim)
        self.scale = output_dim ** 0.5

    def forward(self, query, key, value):
        q = self.query(query)
        k = self.key(key).transpose(-2, -1)
        v = self.value(value)

        attention_scores = torch.matmul(q, k) / self.scale
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_output = torch.matmul(attention_probs, v)
        return attention_output

class DualCrossAttention(nn.Module):
    def __init__(self, dim, output_dim):
        super(DualCrossAttention, self).__init__()
        self.attention1 = ScaledDotProductAttention(dim, output_dim)
        self.attention2 = ScaledDotProductAttention(output_dim, output_dim)

    def forward(self, query1, key1, value1, query2, key2, value2):
        # First attention mechanism
        attended1 = self.attention1(query1, key1, value1)

        # Second attention mechanism
        attended2 = self.attention2(query2, key2, value2)

        return attended1, attended2


class DualCrossAttention2(nn.Module):
    def __init__(self, img_dim=256, gene_dim=138, output_dim=256):
        super(DualCrossAttention2, self).__init__()
        # Attention mechanisms
        self.attention1 = ScaledDotProductAttention(img_dim, output_dim)
        self.attention2 = ScaledDotProductAttention(output_dim, output_dim)

        # Linear layers to adjust dimensions
        self.adjust_gene_dim = nn.Linear(gene_dim, output_dim)

    def forward(self, query1, key1, value1, query2, key2, value2):
        # First attention mechanism
        attended1 = self.attention1(query1, key1, value1)

        # Adjust the dimension of gene features to match the output dimension
        adjusted_gene_features = self.adjust_gene_dim(key2)
        
        # Second attention mechanism
        attended2 = self.attention2(query2, adjusted_gene_features, adjusted_gene_features)

        return attended1, attended2



class TransformerFusion(nn.Module):
    def __init__(self, input_dim, nhead, num_layers, dim_feedforward):
        super(TransformerFusion, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=nhead, dim_feedforward=dim_feedforward)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)

    def forward(self, src):
        return self.transformer_encoder(src)



import torch
import torch.nn as nn
import torch.nn.functional as F

class CoDualCrossAttentionModel(nn.Module):
    def __init__(self, img_dim, gene_dim, co_output_dim, cross_output_dim, nhead, num_layers, dim_feedforward):
        super(CoDualCrossAttentionModel, self).__init__()
        self.co_attention = CoAttention(img_dim, gene_dim, co_output_dim)
        self.dual_cross_attention = DualCrossAttention(co_output_dim, cross_output_dim)
        self.dual_cross_attention2 = DualCrossAttention2(img_dim=256, gene_dim=138, output_dim=256)
        self.transformer_encoder = nn.TransformerEncoderLayer(
            d_model=512, nhead=nhead, dim_feedforward=dim_feedforward)
        self.encoder = nn.TransformerEncoder(self.transformer_encoder, num_layers=num_layers)

    def forward(self, img_features, gene_features):
        # Co-Attention
        coattended_img, coattended_gene = self.co_attention(img_features, gene_features)
        
        
        
        # Dual Cross-Attention - First Stage
        attended_img_to_gene, attended_gene_to_img = self.dual_cross_attention(coattended_img, coattended_gene, coattended_gene, coattended_gene, coattended_img, coattended_img)
        

        # Dual Cross-Attention - Second Stage
        refined_img_features, refined_gene_features = self.dual_cross_attention2(attended_img_to_gene, img_features, img_features, attended_gene_to_img, gene_features, gene_features)

       
        # Transformer Encoder Fusion
        # Combining and reshaping the attended features for the transformer encoder
        combined_features = torch.cat((refined_img_features.unsqueeze(0), refined_gene_features.unsqueeze(0)), dim=2)
        
        
        
        fused_features = self.encoder(combined_features)
        
        
        fused_features = torch.squeeze(fused_features, 0)
       
        return fused_features
