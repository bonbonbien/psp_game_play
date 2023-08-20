import torch
import torch.nn as nn

# # Numerical features used for NN
# num_cols_level2 = ["elapsed_time_log1p","elapsed_time_diff_log1p",
#                   "room_coor_x","room_coor_y"]

# # Categorical features used for NN
# cat_cols = ["event_name","name","fqid","room_fqid","level"]

# https://www.kaggle.com/competitions/predict-student-performance-from-game-play/discussion/420077#2324257
# (1) A separate model for each level group:
# level_group 0-4, I chose a sequence length of 250. 
# For level_group 5-12, the sequence length is 500. 
# And for level_group 13-22, the sequence length is 800.
# (2) Set epochs to 20, choose checkpoints with highest AUC
# (3) The following model is for level group 5-12
# ? the out_size=10 should be replaced with out_size=8 ??


class MeanPooling(nn.Module):
    def __init__(self):
        super(MeanPooling, self).__init__()

    def forward(self, output_hidden_state, attention_mask):
        input_mask_expanded = (~attention_mask).unsqueeze(-1).expand(output_hidden_state.size()).float()
        sum_embeddings = torch.sum(output_hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask
        return mean_embeddings

class PspTransformerModel(nn.Module):
    def __init__(
        self, dropout=0.2,
        input_numerical_size=4,
        input_name_nunique=7, name_embedding_size=12,
        input_event_nunique=12, event_embedding_size = 12,
        input_fqid_nunique=130, fqid_embedding_size=24,
        input_room_fqid_nunique=20, room_fqid_embedding_size=12,
        input_level_nunique=24, level_embedding_size=12,
        categorical_linear_size=120,
        numeraical_linear_size=24,
        model_size=160,
        nhead=16,
        dim_feedforward=480,
        out_size=10):
        super(PspTransformerModel, self).__init__()
        self.name_embedding = nn.Embedding(num_embeddings=input_name_nunique, 
                                           embedding_dim=name_embedding_size)
        self.event_embedding = nn.Embedding(num_embeddings=input_event_nunique, 
                                           embedding_dim=event_embedding_size)
        self.fqid_embedding = nn.Embedding(num_embeddings=input_fqid_nunique, 
                                           embedding_dim=fqid_embedding_size)
        self.room_fqid_embedding = nn.Embedding(num_embeddings=input_room_fqid_nunique, 
                                           embedding_dim=room_fqid_embedding_size)
        self.level_embedding = nn.Embedding(num_embeddings=input_level_nunique, 
                                           embedding_dim=level_embedding_size)
        self.categorical_linear = nn.Sequential(
                nn.Linear(name_embedding_size + event_embedding_size + 
                          fqid_embedding_size + room_fqid_embedding_size + 
                          level_embedding_size, categorical_linear_size),
                nn.LayerNorm(categorical_linear_size)
            )
        self.numerical_linear  = nn.Sequential(
                nn.Linear(input_numerical_size, numeraical_linear_size),
                nn.LayerNorm(numeraical_linear_size)
            )

        self.linear1  = nn.Sequential(
                nn.Linear(categorical_linear_size + numeraical_linear_size, 
                          model_size),
                nn.LayerNorm(model_size),
            )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer = nn.TransformerEncoderLayer(d_model=model_size, 
                                                       nhead=nhead,
                                                       dim_feedforward=dim_feedforward,
                                                       # batch_first=True,
                                                       dropout=dropout),
                                                     num_layers=1)
        self.gru = nn.GRU(model_size, model_size,
                            num_layers=1, 
                            batch_first=True,
                            bidirectional=True)
        self.linear_out  = nn.Sequential(
                nn.Linear(model_size*2, 
                          out_size)
            )
        self.pool = MeanPooling()

    def forward(self, numerical_array, name_array, event_array, fqid_array, room_fqid_array, 
                level_array, mask, mask_for_pooling):

        name_embedding = self.name_embedding(name_array)
        event_embedding = self.event_embedding(event_array)
        fqid_embedding = self.fqid_embedding(fqid_array)
        room_fqid_embedding = self.room_fqid_embedding(room_fqid_array)
        level_embedding = self.level_embedding(level_array)
        categorical_emedding = torch.cat([name_embedding,
                                          event_embedding,
                                          fqid_embedding, 
                                          room_fqid_embedding,
                                          level_embedding
                                          ], axis=2)
        categorical_emedding = self.categorical_linear(categorical_emedding)
        numerical_embedding = self.numerical_linear(numerical_array)
        concat_embedding = torch.cat([categorical_emedding,
                                      numerical_embedding],axis=2)
        concat_embedding = self.linear1(concat_embedding)
        concat_embedding = concat_embedding.permute(1,0,2).contiguous()
        output = self.transformer_encoder(concat_embedding, 
                                          src_key_padding_mask=mask)
        output = output.permute(1,0,2).contiguous()
        output, _ = self.gru(output)
        # mask_for_pooling = mask ?
        output = self.pool(output, mask_for_pooling)
        output = self.linear_out(output)

        return output
